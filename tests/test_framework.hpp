#pragma once

#include <iostream>
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <cmath>
#include <sstream>

namespace silica::test {

class TestRegistry {
public:
    struct TestCase {
        std::string suite_name;
        std::string test_name;
        std::function<void()> func;
    };

    static TestRegistry& instance() {
        static TestRegistry reg;
        return reg;
    }

    void register_test(std::string suite, std::string name, std::function<void()> func) {
        tests_.push_back({std::move(suite), std::move(name), std::move(func)});
    }

    int run_all() {
        int passed = 0;
        int failed = 0;
        auto start_total = std::chrono::high_resolution_clock::now();

        std::cout << "\n======================================================================\n";
        std::cout << "                  Silica OpenCL Test Suite Runner                     \n";
        std::cout << "======================================================================\n\n";

        for (const auto& t : tests_) {
            std::cout << "[ RUN      ] " << t.suite_name << "." << t.test_name << " ... " << std::flush;
            auto start_test = std::chrono::high_resolution_clock::now();
            try {
                t.func();
                auto end_test = std::chrono::high_resolution_clock::now();
                auto ms = std::chrono::duration_cast<std::chrono::microseconds>(end_test - start_test).count();
                std::cout << "\033[32m[PASS]\033[0m (" << (static_cast<double>(ms) / 1000.0) << " ms)\n";
                passed++;
            } catch (const std::exception& ex) {
                std::cout << "\033[31m[FAIL]\033[0m\n";
                std::cout << "             Error: " << ex.what() << "\n";
                failed++;
            } catch (...) {
                std::cout << "\033[31m[FAIL]\033[0m Unknown exception thrown\n";
                failed++;
            }
        }

        auto end_total = std::chrono::high_resolution_clock::now();
        auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total).count();

        std::cout << "\n======================================================================\n";
        std::cout << "TEST RESULTS: " << passed << " Passed, " << failed << " Failed in " << total_ms << " ms\n";
        std::cout << "======================================================================\n\n";

        return (failed == 0) ? 0 : 1;
    }

private:
    std::vector<TestCase> tests_;
};

#define SILICA_TEST(suite, name) \
    static void suite##_##name##_impl(); \
    namespace { \
        struct suite##_##name##_reg { \
            suite##_##name##_reg() { \
                ::silica::test::TestRegistry::instance().register_test(#suite, #name, suite##_##name##_impl); \
            } \
        } suite##_##name##_instance; \
    } \
    static void suite##_##name##_impl()

#define ASSERT_TRUE(cond) \
    do { \
        if (!(cond)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: (" #cond ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)

#define ASSERT_FALSE(cond) ASSERT_TRUE(!(cond))

#define ASSERT_EQ(a, b) \
    do { \
        if ((a) != (b)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: " #a " == " #b " (" << (a) << " vs " << (b) << ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)

#define ASSERT_NEAR(a, b, eps) \
    do { \
        if (std::abs((a) - (b)) > (eps)) { \
            std::ostringstream oss; \
            oss << "Assertion failed: |" #a " - " #b "| <= " #eps " (diff: " << std::abs((a) - (b)) << ") at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(oss.str()); \
        } \
    } while (0)

} // namespace silica::test
