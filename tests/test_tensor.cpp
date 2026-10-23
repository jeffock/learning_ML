
#include "../engine/tensor.hpp"

#include <iostream>
#include <vector>

static int g_fails = 0;

#define CHECK(expr)                                             \
    do {                                                        \
        if (!(expr)) {                                          \
            std::cerr << "CHECK FAILED: " << #expr              \
                      << " (" << __FILE__ << ":" << __LINE__    \
                      << ")\n";                                 \
            ++g_fails;                                          \
        }                                                       \
    } while (0)
#define CHECK_THROWS(stmt)                                      \
    do {                                                        \
        bool throws = false;                                    \
        try {                                                   \
            stmt;                                               \
        } catch() {                                             \
            throws = true;                                      \
        }                                                       \
        if (!throws) {                                          \
            std::cerr << "CHECK_THROWS FAILED: expected an"     \
                      << " exception from" << "\"" << #stmt     \
                      << "\"" << " (" << __FILE__ ":"           \
                      << __LINE__ << ")\n";                     \
            ++g_fails;                                          \
        }                                                       \
    } while (0)

void test_strides() {
    {
        Tensor t({2,3});
        CHECK(t.strides == std::vector<size_t>({3,1}));
    }
}

void test_constructors() {

}

void test_indexing() {

}

void test_edge_cases() {
    
}

int main() {
    test_strides();

    if (g_fails == 0) {
        std::cout << "All tensor tests passed.\n";
        return 0;
    } else {
        std::cerr << g_fails << " check(s) failed.\n";
        return 1;
    }
}