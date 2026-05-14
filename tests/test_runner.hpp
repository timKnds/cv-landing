#pragma once
#include <iostream>
#include <string>
#include <functional>
#include <vector>

struct TestCase {
    std::string name;
    std::function<void()> fn;
};

static std::vector<TestCase>& test_registry() {
    static std::vector<TestCase> cases;
    return cases;
}

#define TEST(name, body) \
    static void test_##name(); \
    static bool _reg_##name = (test_registry().push_back({#name, test_##name}), true); \
    static void test_##name() body

#define ASSERT_TRUE(expr) \
    do { if (!(expr)) { \
        std::fprintf(stderr, "  FAIL: %s:%d  ASSERT_TRUE(%s)\n", __FILE__, __LINE__, #expr); \
        throw std::runtime_error("assertion failed"); \
    }} while(0)

#define ASSERT_NEAR(a, b, tol) \
    do { if (std::abs((a)-(b)) > (tol)) { \
        std::fprintf(stderr, "  FAIL: %s:%d  |%.4f - %.4f| > %.4f\n", \
                     __FILE__, __LINE__, (double)(a), (double)(b), (double)(tol)); \
        throw std::runtime_error("assertion failed"); \
    }} while(0)

inline int run_all_tests() {
    int passed = 0, failed = 0;
    for (const auto& tc : test_registry()) {
        std::printf("[ RUN ] %s\n", tc.name.c_str());
        try {
            tc.fn();
            std::printf("[ OK  ] %s\n", tc.name.c_str());
            ++passed;
        } catch (const std::exception& e) {
            std::printf("[FAIL ] %s — %s\n", tc.name.c_str(), e.what());
            ++failed;
        }
    }
    std::printf("\n%d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
