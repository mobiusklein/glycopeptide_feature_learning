#include<functional>

struct SizeTPairHash {
    size_t operator()(const std::pair<size_t, size_t>& v) const {
        std::hash<size_t> hasher;
        size_t seed = 0;
        seed ^= hasher(v.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        seed ^= hasher(v.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        return seed;
    }
};