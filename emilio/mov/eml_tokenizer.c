/*
 * eml_tokenizer.c -- BPE tokenizer for Qwen2.5 (C89, MOV-only target)
 *
 * Implements GPT-2 style byte-pair encoding:
 *   - byte_to_unicode / unicode_to_byte tables
 *   - Sorted-array + linear scan for merge lookup (no HashMap)
 *   - Sorted-array + binary search for token_to_id lookup
 *   - Special token handling (<|im_start|>, <|im_end|>, <|endoftext|>)
 *   - ChatML template encoding
 */

#include "eml_mov.h"

/* ======================================================================
 * GPT-2 byte <-> unicode mapping
 *
 * Bytes 33..126, 161..172, 174..255 map to themselves as Unicode codepoints.
 * Remaining bytes (0..32, 127..160, 173) map to 256..511 range.
 * ====================================================================== */

void tokenizer_init_byte_tables(Tokenizer *tok)
{
    int b, n;
    n = 256;

    /* clear unicode_to_byte */
    memset(tok->unicode_to_byte, 0, sizeof(tok->unicode_to_byte));

    for (b = 0; b <= 255; b++) {
        int c;
        if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255)) {
            c = b;
        } else {
            c = n;
            n++;
        }
        tok->byte_to_unicode[b] = c;
        if (c < 512) {
            tok->unicode_to_byte[c] = b;
        }
    }
}

/* ======================================================================
 * Build sorted vocab for binary search (string -> id)
 * ====================================================================== */

static void build_sorted_vocab(Tokenizer *tok)
{
    int i;
    tok->sorted_count = tok->vocab_size;
    tok->sorted = (TokenEntry *)malloc((eml_size_t)tok->vocab_size * sizeof(TokenEntry));
    for (i = 0; i < tok->vocab_size; i++) {
        tok->sorted[i].str = tok->vocab[i];
        tok->sorted[i].id = i;
    }
    /* Simple insertion sort (qsort may not be available without stdlib.h) */
    {
        int j;
        for (i = 1; i < tok->sorted_count; i++) {
            TokenEntry tmp = tok->sorted[i];
            j = i - 1;
            while (j >= 0 && strcmp(tok->sorted[j].str, tmp.str) > 0) {
                tok->sorted[j + 1] = tok->sorted[j];
                j--;
            }
            tok->sorted[j + 1] = tmp;
        }
    }
}

static int token_to_id(const Tokenizer *tok, const char *s)
{
    /* Binary search in sorted vocab */
    int lo = 0, hi = tok->sorted_count - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int c = strcmp(tok->sorted[mid].str, s);
        if (c == 0) return tok->sorted[mid].id;
        if (c < 0) lo = mid + 1;
        else hi = mid - 1;
    }
    return 0; /* unknown token */
}

/* ======================================================================
 * Merge lookup: find rank of (left, right) pair.
 * Linear scan -- merges are stored in rank order from the file.
 * Returns -1 if not found.
 * ====================================================================== */

static int find_merge_rank(const Tokenizer *tok, const char *left, const char *right)
{
    int i;
    for (i = 0; i < tok->merges_count; i++) {
        if (strcmp(tok->merges[i].left, left) == 0 &&
            strcmp(tok->merges[i].right, right) == 0) {
            return tok->merges[i].rank;
        }
    }
    return -1;
}

/* ======================================================================
 * Initialize tokenizer lookup tables
 * ====================================================================== */

void tokenizer_init_tables(Tokenizer *tok)
{
    tokenizer_init_byte_tables(tok);
    build_sorted_vocab(tok);

    /* Find special tokens */
    tok->im_start_id = -1;
    tok->im_end_id = -1;
    {
        int id;
        id = token_to_id(tok, "<|im_start|>");
        if (id > 0) tok->im_start_id = id;
        id = token_to_id(tok, "<|im_end|>");
        if (id > 0) tok->im_end_id = id;
    }
}

/* ======================================================================
 * Encode a chunk of text (no special tokens) using BPE
 * ====================================================================== */

/* Helper: encode a unicode codepoint to UTF-8, return number of bytes written */
static int codepoint_to_utf8(int cp, char *out)
{
    if (cp < 0x80) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 0x800) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    } else if (cp < 0x10000) {
        out[0] = (char)(0xE0 | (cp >> 12));
        out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        out[2] = (char)(0x80 | (cp & 0x3F));
        return 3;
    }
    out[0] = (char)(0xF0 | (cp >> 18));
    out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/*
 * BPE encoding of a text chunk:
 *  1. Convert bytes to GPT-2 unicode character strings
 *  2. Start with single-character "symbols"
 *  3. Repeatedly merge the highest-priority adjacent pair
 *  4. Look up each final symbol in the vocab
 */
static int *encode_chunk(const Tokenizer *tok, const char *text, int text_len, int *out_len)
{
    /* symbols: array of strings (each is a BPE token fragment) */
    char **symbols;
    int n_symbols;
    int i, j;
    int *ids;
    int ids_count;

    if (text_len == 0) { *out_len = 0; return (int *)malloc(1); }

    /* Step 1: Convert each byte to its GPT-2 unicode char (as a UTF-8 string) */
    symbols = (char **)malloc((eml_size_t)text_len * sizeof(char *));
    n_symbols = 0;

    for (i = 0; i < text_len; i++) {
        unsigned char b = (unsigned char)text[i];
        int cp = tok->byte_to_unicode[b];
        char buf[8];
        int len;
        char *s;

        len = codepoint_to_utf8(cp, buf);
        s = (char *)malloc((eml_size_t)(len + 1));
        memcpy(s, buf, (eml_size_t)len);
        s[len] = '\0';
        symbols[n_symbols++] = s;
    }

    /* Step 2: BPE merge loop */
    while (n_symbols > 1) {
        int best_rank = 0x7FFFFFFF;
        int best_pos = -1;

        /* Find the lowest-rank merge pair */
        for (i = 0; i < n_symbols - 1; i++) {
            int rank = find_merge_rank(tok, symbols[i], symbols[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_pos = i;
            }
        }

        if (best_pos < 0) break; /* no more merges */

        /* Merge: concatenate symbols[best_pos] and symbols[best_pos+1] */
        {
            int len_a = (int)strlen(symbols[best_pos]);
            int len_b = (int)strlen(symbols[best_pos + 1]);
            char *merged = (char *)malloc((eml_size_t)(len_a + len_b + 1));
            strcpy(merged, symbols[best_pos]);
            strcat(merged, symbols[best_pos + 1]);

            free(symbols[best_pos]);
            free(symbols[best_pos + 1]);
            symbols[best_pos] = merged;

            /* Shift remaining symbols down */
            for (j = best_pos + 1; j < n_symbols - 1; j++) {
                symbols[j] = symbols[j + 1];
            }
            n_symbols--;
        }
    }

    /* Step 3: Convert symbols to token IDs */
    ids = (int *)malloc((eml_size_t)n_symbols * sizeof(int));
    ids_count = 0;
    for (i = 0; i < n_symbols; i++) {
        ids[ids_count++] = token_to_id(tok, symbols[i]);
        free(symbols[i]);
    }
    free(symbols);

    *out_len = ids_count;
    return ids;
}

/* ======================================================================
 * Public encode: handles special tokens, then BPE for text chunks
 * ====================================================================== */

int *tokenizer_encode(const Tokenizer *tok, const char *text, int *out_len)
{
    int text_len;
    int *result;
    int result_len, result_cap;
    const char *remaining;

    /* Special token strings to look for */
    const char *sp_strs[3];
    int sp_ids[3];
    int n_special;

    text_len = (int)strlen(text);
    if (text_len == 0) { *out_len = 0; return (int *)malloc(1); }

    result_cap = text_len + 16;
    result = (int *)malloc((eml_size_t)result_cap * sizeof(int));
    result_len = 0;

    /* Build special token list */
    n_special = 0;
    if (tok->im_start_id >= 0) {
        sp_strs[n_special] = "<|im_start|>";
        sp_ids[n_special] = tok->im_start_id;
        n_special++;
    }
    if (tok->im_end_id >= 0) {
        sp_strs[n_special] = "<|im_end|>";
        sp_ids[n_special] = tok->im_end_id;
        n_special++;
    }
    sp_strs[n_special] = "<|endoftext|>";
    sp_ids[n_special] = tok->eos_id;
    n_special++;

    remaining = text;
    while (*remaining) {
        /* Find earliest special token */
        const char *earliest_ptr = EML_NULL;
        int earliest_id = -1;
        int earliest_sp_len = 0;
        int s;

        for (s = 0; s < n_special; s++) {
            const char *p = remaining;
            int sp_len = (int)strlen(sp_strs[s]);
            /* Manual strstr since we can't include <string.h> */
            while (*p) {
                if (strncmp(p, sp_strs[s], (eml_size_t)sp_len) == 0) {
                    if (earliest_ptr == EML_NULL || p < earliest_ptr) {
                        earliest_ptr = p;
                        earliest_id = sp_ids[s];
                        earliest_sp_len = sp_len;
                    }
                    break;
                }
                p++;
            }
        }

        if (earliest_ptr != EML_NULL) {
            /* Encode text before special token */
            int prefix_len = (int)(earliest_ptr - remaining);
            if (prefix_len > 0) {
                int chunk_len;
                int *chunk = encode_chunk(tok, remaining, prefix_len, &chunk_len);
                int c;
                /* Grow result if needed */
                while (result_len + chunk_len + 1 >= result_cap) {
                    result_cap *= 2;
                    result = (int *)realloc(result, (eml_size_t)result_cap * sizeof(int));
                }
                for (c = 0; c < chunk_len; c++) result[result_len++] = chunk[c];
                free(chunk);
            }
            /* Add special token */
            if (result_len + 1 >= result_cap) {
                result_cap *= 2;
                result = (int *)realloc(result, (eml_size_t)result_cap * sizeof(int));
            }
            result[result_len++] = earliest_id;
            remaining = earliest_ptr + earliest_sp_len;
        } else {
            /* No more special tokens -- encode the rest */
            int rest_len = (int)strlen(remaining);
            int chunk_len;
            int *chunk = encode_chunk(tok, remaining, rest_len, &chunk_len);
            int c;
            while (result_len + chunk_len >= result_cap) {
                result_cap *= 2;
                result = (int *)realloc(result, (eml_size_t)result_cap * sizeof(int));
            }
            for (c = 0; c < chunk_len; c++) result[result_len++] = chunk[c];
            free(chunk);
            break;
        }
    }

    *out_len = result_len;
    return result;
}

/* ======================================================================
 * Decode: token IDs -> text
 * ====================================================================== */

char *tokenizer_decode(const Tokenizer *tok, const int *ids, int n)
{
    /* First pass: compute total output size */
    int total = 0;
    int i;
    char *out;
    int pos;

    for (i = 0; i < n; i++) {
        const char *token;
        if (ids[i] < 0 || ids[i] >= tok->vocab_size) continue;
        token = tok->vocab[ids[i]];
        /* Skip special tokens */
        if (token[0] == '<' && token[1] == '|') continue;
        /* Each GPT-2 unicode char in the token maps to 1 byte */
        {
            const char *p = token;
            while (*p) {
                /* Decode UTF-8 codepoint */
                unsigned char c = (unsigned char)*p;
                int cp;
                if (c < 0x80) {
                    cp = c; p++;
                } else if ((c & 0xE0) == 0xC0) {
                    cp = (c & 0x1F) << 6;
                    cp |= ((unsigned char)p[1]) & 0x3F;
                    p += 2;
                } else if ((c & 0xF0) == 0xE0) {
                    cp = (c & 0x0F) << 12;
                    cp |= (((unsigned char)p[1]) & 0x3F) << 6;
                    cp |= ((unsigned char)p[2]) & 0x3F;
                    p += 3;
                } else {
                    cp = (c & 0x07) << 18;
                    cp |= (((unsigned char)p[1]) & 0x3F) << 12;
                    cp |= (((unsigned char)p[2]) & 0x3F) << 6;
                    cp |= ((unsigned char)p[3]) & 0x3F;
                    p += 4;
                }
                total++;  /* each codepoint maps to 1 byte */
            }
        }
    }

    out = (char *)malloc((eml_size_t)(total + 1));
    pos = 0;

    for (i = 0; i < n; i++) {
        const char *token;
        if (ids[i] < 0 || ids[i] >= tok->vocab_size) continue;
        token = tok->vocab[ids[i]];
        if (token[0] == '<' && token[1] == '|') continue;
        {
            const char *p = token;
            while (*p) {
                unsigned char c = (unsigned char)*p;
                int cp;
                if (c < 0x80) {
                    cp = c; p++;
                } else if ((c & 0xE0) == 0xC0) {
                    cp = (c & 0x1F) << 6;
                    cp |= ((unsigned char)p[1]) & 0x3F;
                    p += 2;
                } else if ((c & 0xF0) == 0xE0) {
                    cp = (c & 0x0F) << 12;
                    cp |= (((unsigned char)p[1]) & 0x3F) << 6;
                    cp |= ((unsigned char)p[2]) & 0x3F;
                    p += 3;
                } else {
                    cp = (c & 0x07) << 18;
                    cp |= (((unsigned char)p[1]) & 0x3F) << 12;
                    cp |= (((unsigned char)p[2]) & 0x3F) << 6;
                    cp |= ((unsigned char)p[3]) & 0x3F;
                    p += 4;
                }
                /* Map unicode codepoint back to byte */
                if (cp < 512) {
                    out[pos++] = (char)tok->unicode_to_byte[cp];
                } else {
                    out[pos++] = '?';
                }
            }
        }
    }
    out[pos] = '\0';
    return out;
}

/* ======================================================================
 * Encode with ChatML template
 * ====================================================================== */

int *tokenizer_encode_chat(const Tokenizer *tok, const char *msg, int *out_len)
{
    /* Build ChatML string:
     * <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
     * <|im_start|>user\n{msg}<|im_end|>\n
     * <|im_start|>assistant\n
     */
    int msg_len = (int)strlen(msg);
    int buf_len = msg_len + 256;
    char *buf = (char *)malloc((eml_size_t)buf_len);
    int *result;

    strcpy(buf, "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n");
    strcat(buf, msg);
    strcat(buf, "<|im_end|>\n<|im_start|>assistant\n");

    result = tokenizer_encode(tok, buf, out_len);
    free(buf);
    return result;
}

/* ======================================================================
 * Free tokenizer resources
 * ====================================================================== */

void tokenizer_free(Tokenizer *tok)
{
    int i;
    if (tok->vocab) {
        for (i = 0; i < tok->vocab_size; i++) free(tok->vocab[i]);
        free(tok->vocab);
    }
    free(tok->sorted);
    if (tok->merges) {
        for (i = 0; i < tok->merges_count; i++) {
            free(tok->merges[i].left);
            free(tok->merges[i].right);
        }
        free(tok->merges);
    }
}
