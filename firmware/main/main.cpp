// Measurement firmware.
//
// Protocol (little-endian, one request at a time):
//   'H'                       -> text line "BENCH READY ...\n"
//   'S' + N_FEATURES float32  -> 'R' int32 label, uint32 median cycles, uint32 min cycles, uint32 timed calls
//   'E'                       -> 'M' uint32 stack size, uint32 stack high-water mark (bytes),
//                                    uint32 free heap, uint32 minimum free heap

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>

#include "esp_cpu.h"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "sdkconfig.h"

#include "link.h"
#include "model_info.h"

extern "C" int bench_predict(const float *x);

namespace {

constexpr int kRepeats = CONFIG_BENCH_REPEATS;

struct __attribute__((packed)) SampleReply {
    uint8_t tag;
    int32_t label;
    uint32_t cycles_median;
    uint32_t cycles_min;
    uint32_t timed_calls;
};

struct __attribute__((packed)) EndReply {
    uint8_t tag;
    uint32_t stack_size;
    uint32_t stack_free_min;
    uint32_t heap_free;
    uint32_t heap_free_min;
};

float g_x[BENCH_N_FEATURES];
volatile int g_sink;

uint32_t timed_call(int *label)
{
    esp_cpu_cycle_count_t t0 = esp_cpu_get_cycle_count();
    int y = bench_predict(g_x);
    esp_cpu_cycle_count_t t1 = esp_cpu_get_cycle_count();
    g_sink = y;
    *label = y;
    return (uint32_t)(t1 - t0);
}

void send_hello()
{
    char line[200];
    int n = snprintf(line, sizeof line,
                     "BENCH READY id=%s target=%s mhz=%d nf=%d nc=%d warmup=%d repeats=%d\n",
                     BENCH_MODEL_ID, CONFIG_IDF_TARGET, CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ,
                     BENCH_N_FEATURES, BENCH_N_CLASSES, CONFIG_BENCH_WARMUP, kRepeats);
    link_write(line, (size_t)n);
}

void handle_sample()
{
    link_read(g_x, sizeof g_x);

    int label = 0;
    uint32_t cycles[kRepeats];
    int timed = 0;

    // The first call decides whether the model is too slow to repeat.
    uint32_t first = timed_call(&label);
    if (first > (uint32_t)CONFIG_BENCH_SLOW_CYCLES) {
        cycles[timed++] = first;
    } else {
        for (int i = 1; i < CONFIG_BENCH_WARMUP; i++) {
            timed_call(&label);
        }
        for (; timed < kRepeats; timed++) {
            cycles[timed] = timed_call(&label);
        }
    }

    std::sort(cycles, cycles + timed);
    SampleReply reply = {'R', label, cycles[timed / 2], cycles[0], (uint32_t)timed};
    link_write(&reply, sizeof reply);
}

void handle_end()
{
    EndReply reply = {
        'M',
        (uint32_t)CONFIG_BENCH_TASK_STACK,
        (uint32_t)uxTaskGetStackHighWaterMark(NULL),  // bytes: StackType_t is uint8_t
        (uint32_t)heap_caps_get_free_size(MALLOC_CAP_8BIT),
        (uint32_t)heap_caps_get_minimum_free_size(MALLOC_CAP_8BIT),
    };
    link_write(&reply, sizeof reply);
}

void bench_task(void *)
{
    send_hello();
    for (;;) {
        uint8_t cmd = 0;
        link_read(&cmd, 1);
        switch (cmd) {
        case 'H': send_hello(); break;
        case 'S': handle_sample(); break;
        case 'E': handle_end(); break;
        default: break;
        }
    }
}

}  // namespace

extern "C" void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_NONE);
    link_init();
    // Run on the last core so that the measurement does not share a core with system tasks
    // on dual-core chips.
    xTaskCreatePinnedToCore(bench_task, "bench", CONFIG_BENCH_TASK_STACK, NULL,
                            tskIDLE_PRIORITY + 5, NULL, CONFIG_FREERTOS_NUMBER_OF_CORES - 1);
}
