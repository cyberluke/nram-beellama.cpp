// llama-dedup-manifest.cpp — persistent layer manifest (see header).

#include "llama-dedup-manifest.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#ifdef _WIN32
#include <direct.h>
#endif
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Compact public-domain SHA-256 (Brad Conte-style, self-contained). The fork's
// vendor/hash wrapper isn't linked into llama.dll, so we carry the digest here:
// the manifest needs a durable, collision-resistant identity for layer blobs.
// ---------------------------------------------------------------------------

typedef struct {
    unsigned char data[64];
    uint32_t datalen;
    unsigned long long bitlen;
    uint32_t state[8];
} bee_sha256_ctx;

static const uint32_t bee_sha256_k[64] = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

#define BEE_RROTR(x,n) (((x)>>(n))|((x)<<(32-(n))))
static void bee_sha256_transform(bee_sha256_ctx *ctx, const unsigned char data[]) {
    uint32_t a,b,c,d,e,f,g,h,i,j,t1,t2,m[64];
    for (i=0,j=0;i<16;++i, j+=4)
        m[i]=(data[j]<<24)|(data[j+1]<<16)|(data[j+2]<<8)|(data[j+3]);
    for (;i<64;++i)
        m[i]=((BEE_RROTR(m[i-2],17))^(BEE_RROTR(m[i-2],19))^(m[i-2]>>10))+
             (m[i-7])+((BEE_RROTR(m[i-15],7))^(BEE_RROTR(m[i-15],18))^(m[i-15]>>3))+
             (m[i-16]);
    a=ctx->state[0];b=ctx->state[1];c=ctx->state[2];d=ctx->state[3];
    e=ctx->state[4];f=ctx->state[5];g=ctx->state[6];h=ctx->state[7];
    for (i=0;i<64;++i) {
        t1=h+((BEE_RROTR(e,6))^(BEE_RROTR(e,11))^(BEE_RROTR(e,25)))+((e&f)^(~e&g))+bee_sha256_k[i]+m[i];
        t2=((BEE_RROTR(a,2))^(BEE_RROTR(a,13))^(BEE_RROTR(a,22)))+((a&b)^(a&c)^(b&c));
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    ctx->state[0]+=a;ctx->state[1]+=b;ctx->state[2]+=c;ctx->state[3]+=d;
    ctx->state[4]+=e;ctx->state[5]+=f;ctx->state[6]+=g;ctx->state[7]+=h;
}

static void bee_sha256_init(bee_sha256_ctx *ctx) {
    ctx->datalen=0; ctx->bitlen=0;
    ctx->state[0]=0x6a09e667;ctx->state[1]=0xbb67ae85;
    ctx->state[2]=0x3c6ef372;ctx->state[3]=0xa54ff53a;
    ctx->state[4]=0x510e527f;ctx->state[5]=0x9b05688c;
    ctx->state[6]=0x1f83d9ab;ctx->state[7]=0x5be0cd19;
}

static void bee_sha256_update(bee_sha256_ctx *ctx, const unsigned char data[], size_t len) {
    for (size_t i=0;i<len;++i) {
        ctx->data[ctx->datalen]=data[i]; ctx->datalen++;
        if (ctx->datalen==64) { bee_sha256_transform(ctx,ctx->data); ctx->bitlen+=512; ctx->datalen=0; }
    }
}

static void bee_sha256_final(bee_sha256_ctx *ctx, unsigned char hash[]) {
    uint32_t i=ctx->datalen;
    ctx->data[i++]=0x80;
    if (i>56) { while(i<64) ctx->data[i++]=0; bee_sha256_transform(ctx,ctx->data); i=0; }
    while(i<56) ctx->data[i++]=0;
    ctx->bitlen += ctx->datalen*8;
    ctx->data[63]=ctx->bitlen; ctx->data[62]=ctx->bitlen>>8;
    ctx->data[61]=ctx->bitlen>>16; ctx->data[60]=ctx->bitlen>>24;
    ctx->data[59]=ctx->bitlen>>32; ctx->data[58]=ctx->bitlen>>40;
    ctx->data[57]=ctx->bitlen>>48; ctx->data[56]=ctx->bitlen>>56;
    bee_sha256_transform(ctx,ctx->data);
    for (i=0;i<4;++i) {
        hash[i]   =(ctx->state[0]>>(24-i*8))&0xff;
        hash[i+4] =(ctx->state[1]>>(24-i*8))&0xff;
        hash[i+8] =(ctx->state[2]>>(24-i*8))&0xff;
        hash[i+12]=(ctx->state[3]>>(24-i*8))&0xff;
        hash[i+16]=(ctx->state[4]>>(24-i*8))&0xff;
        hash[i+20]=(ctx->state[5]>>(24-i*8))&0xff;
        hash[i+24]=(ctx->state[6]>>(24-i*8))&0xff;
        hash[i+28]=(ctx->state[7]>>(24-i*8))&0xff;
    }
}

static std::string bee_sha256_hex(const uint8_t * data, size_t len) {
    bee_sha256_ctx c; bee_sha256_init(&c); bee_sha256_update(&c, data, len);
    unsigned char h[32]; bee_sha256_final(&c, h);
    static const char hex[] = "0123456789abcdef";
    std::string out; out.reserve(64);
    for (int i=0;i<32;++i){ out+=hex[h[i]>>4]; out+=hex[h[i]&0xf]; }
    return out;
}
#undef BEE_RROTR

// ---------------------------------------------------------------------------
// Very small JSON serializer/parser for the manifest index. Kept dependency-free
// (the fork already ships vendor/nlohmann but this keeps the module self-contained
// and trivially reviewable for an experimental feature).
// ---------------------------------------------------------------------------

static std::string json_escape(const std::string & s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:   o += c;      break;
        }
    }
    return o;
}

// ---------------------------------------------------------------------------

static std::string g_cache_dir = "beehive-cache";

void llama_dedup_manifest::set_cache_dir(const std::string & dir) {
    g_cache_dir = dir;
}

std::string llama_dedup_manifest::blob_path(const std::string & sha256) {
    return g_cache_dir + "/" + sha256 + ".blob";
}

std::string manifest_path() {
    return g_cache_dir + "/manifest.json";
}

void llama_dedup_manifest::load() {
    loaded_ = false;
    entries_.clear();
    ensure_index_loaded();
}

void llama_dedup_manifest::ensure_index_loaded() {
    if (loaded_) return;
    loaded_ = true;
    std::ifstream in(manifest_path().c_str());
    if (!in.good()) return; // no manifest yet
    std::stringstream buf;
    buf << in.rdbuf();
    parse_json_entries(buf.str());
}

// Very small tolerant parser for our own {"layers":[{...}]} index.
void llama_dedup_manifest::parse_json_entries(const std::string & text) {
    // find each object block and pull the fields we need (order-insensitive).
    size_t pos = 0;
    while ((pos = text.find('{', pos)) != std::string::npos) {
        size_t end = text.find('}', pos);
        if (end == std::string::npos) break;
        std::string blk = text.substr(pos, end - pos + 1);
        pos = end + 1;

        llama_dedup_manifest_entry e{};
        size_t n = 0;
        std::string sha, blob;
        auto field = [&](const char * key) -> std::string {
            std::string k = "\"" + std::string(key) + "\":\"";
            size_t i = blk.find(k);
            if (i == std::string::npos) return {};
            i += k.size();
            size_t j = blk.find('"', i);
            return blk.substr(i, j == std::string::npos ? std::string::npos : j - i);
        };
        e.name = field("name");
        if (e.name.empty()) continue;
        const std::string t = field("type");
        // map a small set of type ids stored in the index
        static const char * ids[] = { "F32","F16","BF16","Q4_0","Q4_1","Q5_0","Q5_1","Q8_0","Q8_1","Q2_K","Q3_K","Q4_K","Q5_K","Q6_K","Q8_K","IQ2_XXS","IQ2_XS","IQ3_XXS","TQ2_0","TQ1_0","Q4_0S" };
        // parse ne on the next fields; for robust cross-version we store ne as a
        // "dims":"a,b,c,d" string.
        std::string dims = field("dims");
        int64_t ne4[4] = {0,0,0,0};
        if (!dims.empty()) {
            std::stringstream ss(dims); std::string tok; int idx=0;
            while (std::getline(ss, tok, ',')) { if(idx<4) ne4[idx++]=atoll(tok.c_str()); }
        }
        e.sha256 = field("sha256");
        e.blob   = field("blob");
        e.bytes  = strtoull(field("bytes").c_str(), nullptr, 10);
        if (e.sha256.empty()) continue;
        // map the numeric ggml_type id we persist (fall back to F16 if absent).
        auto tid = blk.find("\"type_id\":");
        e.type = GGML_TYPE_F16;
        if (tid != std::string::npos) {
            e.type = (ggml_type) atoi(blk.c_str() + tid + 10);
        }
        // native-domain fingerprint (Tier-3), 16 hex chars.
        e.fp = strtoull(field("fp").c_str(), nullptr, 16);
        memcpy(e.ne, ne4, sizeof(ne4));
        entries_[e.name + "|" + e.sha256] = e;
    }
}

const llama_dedup_manifest_entry * llama_dedup_manifest::lookup(
        const char * name, ggml_type type, const int64_t ne[4], const std::string & sha256) const {
    (void) type; (void) ne;
    auto it = entries_.find(std::string(name) + "|" + sha256);
    if (it == entries_.end()) return nullptr;
    return &it->second;
}

const llama_dedup_manifest_entry * llama_dedup_manifest::persist(
        const char * name, ggml_type type, const int64_t ne[4],
        const uint8_t * data, size_t len, uint64_t fp) {
    // If a matching content blob already exists, do not rewrite it (delta == none).
    const std::string sha = bee_sha256_hex(data, len);
    auto * existing = lookup(name, type, ne, sha);
    if (existing != nullptr) {
        return existing;
    }
    // Ensure the cache dir exists before writing the blob.
#ifdef _WIN32
    _mkdir(g_cache_dir.c_str());
#else
    mkdir(g_cache_dir.c_str(), 0755);
#endif
    const std::string bp = blob_path(sha);
    std::ofstream out(bp.c_str(), std::ios::binary);
    if (!out.good()) {
        return nullptr;
    }
    out.write((const char *) data, (std::streamsize) len);
    out.close();

    llama_dedup_manifest_entry e;
    e.name = name;
    e.type = type;
    for (int i = 0; i < 4; i++) e.ne[i] = ne[i];
    e.sha256 = sha;
    e.blob   = sha + ".blob";
    e.bytes  = len;
    e.fp     = fp;
    entries_[e.name + "|" + e.sha256] = e;
    return &entries_[e.name + "|" + e.sha256];
}

void llama_dedup_manifest::flush() {
    std::ofstream out(manifest_path().c_str());
    if (!out.good()) return;
    out << "{\"layer_count\": " << entries_.size() << ", \"layers\": [";
    bool first = true;
    for (const auto & kv : entries_) {
        const auto & e = kv.second;
        if (!first) out << ",";
        first = false;
        char dims[80];
        snprintf(dims, sizeof(dims), "%lld,%lld,%lld,%lld",
            (long long) e.ne[0], (long long) e.ne[1], (long long) e.ne[2], (long long) e.ne[3]);
        char fphex[17];
        snprintf(fphex, sizeof(fphex), "%016llx", (unsigned long long) e.fp);
        out << "{"
            << "\"name\":\"" << json_escape(e.name) << "\","
            << "\"dims\":\"" << dims << "\","
            << "\"type_id\": " << (unsigned int) e.type << ","
            << "\"sha256\":\"" << e.sha256 << "\","
            << "\"fp\":\"" << fphex << "\","
            << "\"blob\":\"" << e.blob << "\","
            << "\"bytes\": " << e.bytes
            << "}";
    }
    out << "]}\n";
}