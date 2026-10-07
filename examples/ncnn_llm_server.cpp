// ncnn_llm-server —— ncnn_llm 的 OpenAI 兼容 HTTP 服务
//
// 让 ncnn_llm 可以被任何 OpenAI 兼容客户端/基准工具（如 llama-benchy）直接评估，
// 也可作为常驻推理服务使用。
//
// 目的：让 ncnn_llm 能被「非自研」的评估工具测量（llama-benchy 只打 OpenAI 兼容端点）。
//
// 诚实性要求（重要）：
//   * generate 的回调每收到一段文本，立即 write() + 不缓冲地送出（绝不攒批），
//     这样 llama-benchy 测到的 TTFT / 生成速率就是真实推理速度；
//   * usage 里的 prompt_tokens 用运行时的真实计数（ctx->position_id），
//     completion_tokens 用真实回调次数；
//   * 不做任何"计时美化"（不预生成、不缓存、不重放）。
//
// 端点：
//   GET  /v1/models            → 模型列表（llama-benchy 用它自动探测模型名）
//   POST /v1/chat/completions  → 支持 stream=true(SSE 逐段) 与 stream=false(一次性)
//   GET  /health               → 健康检查
//
// 用法：
//   ncnn_llm_server --model <dir> [--threads N] [--port P] [--bf16] [--name NAME]
//
// 注：在 SpacemiT K3 上若需要 IME2/A100 簇，请用 ai-run 启动（平台相关，与代码无关）。

#include "ncnn_llm_gpt.h"
#include "net.h"
#include <cpu.h>   // set_omp_num_threads / set_omp_dynamic（ncnn 内部头，k3bench 同样用法）

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// ---------------------------------------------------------------- 极简 JSON 取值
// 只处理本场景需要的键，避免引入依赖。返回值做了最小转义处理。
static std::string json_escape(const std::string& s)
{
    std::string o;
    o.reserve(s.size() + 16);
    for (unsigned char c : s)
    {
        switch (c)
        {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        default:
            if (c < 0x20)
            {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                o += buf;
            }
            else
            {
                o += (char)c;
            }
        }
    }
    return o;
}

// 找 "key" 之后的字符串值（含 \" 转义）
static bool json_get_string(const std::string& body, const std::string& key, std::string& out)
{
    const std::string pat = "\"" + key + "\"";
    size_t p = body.find(pat);
    if (p == std::string::npos)
        return false;
    p = body.find(':', p + pat.size());
    if (p == std::string::npos)
        return false;
    p = body.find('"', p);
    if (p == std::string::npos)
        return false;
    std::string s;
    for (size_t i = p + 1; i < body.size(); i++)
    {
        const char c = body[i];
        if (c == '\\' && i + 1 < body.size())
        {
            const char n = body[++i];
            if (n == 'n') s += '\n';
            else if (n == 't') s += '\t';
            else if (n == 'r') s += '\r';
            else if (n == 'u')
            {
                // 仅支持 BMP 基本情形：解析 4 位十六进制
                if (i + 4 < body.size())
                {
                    unsigned cp = (unsigned)strtoul(body.substr(i + 1, 4).c_str(), nullptr, 16);
                    i += 4;
                    if (cp < 0x80)
                    {
                        s += (char)cp;
                    }
                    else if (cp < 0x800)
                    {
                        s += (char)(0xC0 | (cp >> 6));
                        s += (char)(0x80 | (cp & 0x3F));
                    }
                    else
                    {
                        s += (char)(0xE0 | (cp >> 12));
                        s += (char)(0x80 | ((cp >> 6) & 0x3F));
                        s += (char)(0x80 | (cp & 0x3F));
                    }
                }
            }
            else
            {
                s += n;
            }
        }
        else if (c == '"')
        {
            out = s;
            return true;
        }
        else
        {
            s += c;
        }
    }
    return false;
}

static bool json_get_int(const std::string& body, const std::string& key, long& out)
{
    const std::string pat = "\"" + key + "\"";
    size_t p = body.find(pat);
    if (p == std::string::npos)
        return false;
    p = body.find(':', p + pat.size());
    if (p == std::string::npos)
        return false;
    out = strtol(body.c_str() + p + 1, nullptr, 10);
    return true;
}

static bool json_get_bool(const std::string& body, const std::string& key, bool def)
{
    const std::string pat = "\"" + key + "\"";
    size_t p = body.find(pat);
    if (p == std::string::npos)
        return def;
    p = body.find(':', p + pat.size());
    if (p == std::string::npos)
        return def;
    const char* q = body.c_str() + p + 1;
    while (*q == ' ')
        q++;
    return strncmp(q, "true", 4) == 0;
}

// 取最后一条 user 消息的 content（llama-benchy 只发单轮 prompt）
static std::string extract_user_content(const std::string& body)
{
    std::string best;
    size_t p = 0;
    while (true)
    {
        const size_t r = body.find("\"role\"", p);
        if (r == std::string::npos)
            break;
        std::string role;
        if (json_get_string(body.substr(r, 200), "role", role) && role == "user")
        {
            const size_t c = body.find("\"content\"", r);
            if (c != std::string::npos)
            {
                std::string content;
                if (json_get_string(body.substr(c, body.size() - c), "content", content))
                    best = content;
            }
        }
        p = r + 6;
    }
    return best;
}

// ---------------------------------------------------------------- 全局模型
static ncnn_llm_gpt* g_model = nullptr;
static std::mutex g_gen_mutex;  // 运行时非线程安全：串行化生成（与真实单实例服务一致）
static std::string g_model_name = "qwen3-0.6b";
static int g_default_max_tokens = 128;

static void send_all(int fd, const std::string& s)
{
    size_t off = 0;
    while (off < s.size())
    {
        const ssize_t n = ::send(fd, s.data() + off, s.size() - off, MSG_NOSIGNAL);
        if (n <= 0)
            return;
        off += (size_t)n;
    }
}

static void send_headers(int fd, const char* ctype)
{
    char hdr[256];
    snprintf(hdr, sizeof(hdr),
             "HTTP/1.1 200 OK\r\nContent-Type: %s\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n", ctype);
    send_all(fd, hdr);
}

static void handle_models(int fd)
{
    send_headers(fd, "application/json");
    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"object\":\"list\",\"data\":[{\"id\":\"%s\",\"object\":\"model\",\"created\":0,\"owned_by\":\"ncnn_llm\"}]}",
             g_model_name.c_str());
    send_all(fd, buf);
}

static void handle_chat(int fd, const std::string& body)
{
    std::string prompt = extract_user_content(body);
    if (prompt.empty())
    {
        send_headers(fd, "application/json");
        send_all(fd, "{\"error\":{\"message\":\"no user message\",\"type\":\"invalid_request_error\"}}");
        return;
    }
    long max_tokens = g_default_max_tokens;
    json_get_int(body, "max_tokens", max_tokens);
    if (max_tokens <= 0 || max_tokens > 4096)
        max_tokens = g_default_max_tokens;
    /* OpenAI 约定：未显式请求 stream 时按**非流式**返回（llama-benchy 的 warmup/延迟测量
     * 就是这样发的；此前默认 true 会让它拿到 text/event-stream 而报 mimetype 错误）。 */
    const bool stream = json_get_bool(body, "stream", false);

    const std::string id = "chatcmpl-ncnn";

    if (stream)
        send_headers(fd, "text/event-stream");

    // ---- 真实推理：prefill 后逐段流式送出 ----
    std::lock_guard<std::mutex> lk(g_gen_mutex);

    auto ctx = g_model->prefill(prompt);
    if (!ctx)
    {
        if (stream)
        {
            send_all(fd, "data: {\"error\":\"prefill failed\"}\n\n");
            send_all(fd, "data: [DONE]\n\n");
        }
        else
        {
            send_headers(fd, "application/json");
            send_all(fd, "{\"error\":{\"message\":\"prefill failed\"}}");
        }
        return;
    }
    const int prompt_tokens = ctx->position_id;

    GenerateConfig cfg;
    cfg.max_new_tokens = (int)max_tokens;
    cfg.do_sample = 0;  // 贪心：与 k3bench 口径一致、可复现
    cfg.temperature = 0.0f;
    cfg.top_k = 1;
    cfg.top_p = 1.0f;
    cfg.repetition_penalty = 1.0f;
    cfg.enable_thinking = true;

    int completion_tokens = 0;
    std::string full;
    if (stream)
    {
        // 首个 role chunk（OpenAI 兼容）
        {
            char b[256];
            snprintf(b, sizeof(b),
                     "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"},\"finish_reason\":null}]}\n\n",
                     id.c_str(), (long)time(nullptr), g_model_name.c_str());
            send_all(fd, b);
        }
        g_model->generate(ctx, cfg, [&](const std::string& piece) {
            completion_tokens++;
            full += piece;
            // 每段立即送出（不缓冲、不批量）
            char pre[256];
            snprintf(pre, sizeof(pre), "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"",
                     id.c_str(), (long)time(nullptr), g_model_name.c_str());
            std::string ev = pre;
            ev += json_escape(piece);
            ev += "\"},\"finish_reason\":null}]}\n\n";
            send_all(fd, ev);
        });
        char fin[512];
        snprintf(fin, sizeof(fin),
                 "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],"
                 "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n",
                 id.c_str(), prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
        send_all(fd, fin);
        /* OpenAI 流式约定：usage 放在**独立的、choices 为空**的 chunk 里。
         * llama-benchy 只从这里读 prompt/completion token 数（缺失时会打印
         * "no usage stats found" 并采不到结果）。 */
        {
            char usg[384];
            snprintf(usg, sizeof(usg),
                     "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\",\"created\":%ld,\"model\":\"%s\",\"choices\":[],"
                     "\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}\n\n",
                     id.c_str(), (long)time(nullptr), g_model_name.c_str(),
                     prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
            send_all(fd, usg);
        }
        send_all(fd, "data: [DONE]\n\n");
    }
    else
    {
        g_model->generate(ctx, cfg, [&](const std::string& piece) {
            completion_tokens++;
            full += piece;
        });
        send_headers(fd, "application/json");
        char head[512];
        snprintf(head, sizeof(head),
                 "{\"id\":\"%s\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"",
                 id.c_str());
        std::string resp = head;
        resp += json_escape(full);
        char tail[256];
        snprintf(tail, sizeof(tail),
                 "\"},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":%d,\"completion_tokens\":%d,\"total_tokens\":%d}}",
                 prompt_tokens, completion_tokens, prompt_tokens + completion_tokens);
        resp += tail;
        send_all(fd, resp);
    }
}

static void serve_client(int fd)
{
    // 读请求头
    std::string req;
    char buf[8192];
    size_t header_end = std::string::npos;
    while (true)
    {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0)
            break;
        req.append(buf, (size_t)n);
        header_end = req.find("\r\n\r\n");
        if (header_end != std::string::npos)
            break;
        if (req.size() > 1 << 20)
            break;
    }
    if (header_end == std::string::npos)
    {
        ::close(fd);
        return;
    }

    // 请求行
    const size_t sp1 = req.find(' ');
    const size_t sp2 = req.find(' ', sp1 + 1);
    const std::string method = req.substr(0, sp1);
    const std::string path = req.substr(sp1 + 1, sp2 - sp1 - 1);

    // 读取 body（按 Content-Length）
    size_t content_length = 0;
    {
        const size_t cl = req.find("Content-Length:");
        if (cl != std::string::npos && cl < header_end)
            content_length = strtoul(req.c_str() + cl + 15, nullptr, 10);
    }
    std::string body = req.substr(header_end + 4);
    while (body.size() < content_length)
    {
        const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
        if (n <= 0)
            break;
        body.append(buf, (size_t)n);
    }

    /* 路径既接受 OpenAI 标准的 "/v1/xxx"，也接受不带前缀的 "/xxx"
     * （不同客户端拼接方式不同，例如 llama-benchy 会拼 base_url + "/chat/completions"）。 */
    auto match = [&](const char* tail) {
        const std::string t = tail;
        return path.rfind(t, 0) == 0 || (path.size() > 3 && path.compare(0, 3, "/v1") == 0 && path.rfind("/v1" + t, 0) == 0);
    };

    if (match("/models"))
    {
        handle_models(fd);
    }
    else if (match("/chat/completions"))
    {
        handle_chat(fd, body);
    }
    else if (match("/health"))
    {
        send_headers(fd, "application/json");
        send_all(fd, "{\"status\":\"ok\"}");
    }
    else if (match("/completions"))
    {
        // 极简：把 prompt 当单轮对话处理
        std::string p;
        json_get_string(body, "prompt", p);
        std::string wrapped = "{\"messages\":[{\"role\":\"user\",\"content\":\"" + json_escape(p) + "\"}],\"stream\":false}";
        handle_chat(fd, wrapped);
    }
    else
    {
        const char* hdr = "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n";
        send_all(fd, hdr);
        send_all(fd, "{\"error\":{\"message\":\"not found\",\"type\":\"invalid_request_error\"}}");
    }

    ::close(fd);
}

int main(int argc, char** argv)
{
    std::string model_path;
    int threads = 8;
    int port = 8080;
    bool bf16 = false;

    for (int i = 1; i < argc; i++)
    {
        const std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : std::string(); };
        if (a == "--model") model_path = next();
        else if (a == "--threads") threads = atoi(next().c_str());
        else if (a == "--port") port = atoi(next().c_str());
        else if (a == "--name") g_model_name = next();
        else if (a == "--bf16") bf16 = true;
        else if (a == "--max-tokens") g_default_max_tokens = atoi(next().c_str());
    }
    if (model_path.empty())
    {
        fprintf(stderr, "usage: ncnn_llm_server --model <dir> [--threads N] [--port P] [--bf16] [--name NAME]\n");
        return 1;
    }

    ncnn::set_omp_dynamic(0);
    ncnn::set_omp_num_threads(threads);

    fprintf(stderr, "[ncnn_llm-server] loading %s (threads=%d bf16=%d)\n", model_path.c_str(), threads, (int)bf16);
    g_model = new ncnn_llm_gpt(model_path, false, threads, 0, bf16);
    fprintf(stderr, "[ncnn_llm-server] ready on port %d\n", port);

    const int srv = ::socket(AF_INET, SOCK_STREAM, 0);
    if (srv < 0)
    {
        perror("socket");
        return 1;
    }
    int one = 1;
    setsockopt(srv, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port);
    if (::bind(srv, (sockaddr*)&addr, sizeof(addr)) < 0)
    {
        perror("bind");
        return 1;
    }
    if (::listen(srv, 64) < 0)
    {
        perror("listen");
        return 1;
    }
    fprintf(stderr, "[ncnn_llm-server] listening\n");

    while (true)
    {
        const int fd = ::accept(srv, nullptr, nullptr);
        if (fd < 0)
            continue;
        int nd = 1;
        setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd));  // 低延迟：不等 Nagle
        /* 在主线程串行处理请求：ncnn 运行时（尤其 LoongArch LASX 构建）不保证可被
         * 任意线程调用（实测龙芯上每请求开线程会在生成阶段崩溃；K3 上正常）。
         * 串行处理与"单实例服务"的真实语义一致，也不影响基准测量。 */
        serve_client(fd);
    }
    return 0;
}
