#include "link.h"

#include <stdint.h>

#include "freertos/FreeRTOS.h"
#include "sdkconfig.h"

#define RX_BUFFER 8192
#define TX_BUFFER 1024

#if CONFIG_BENCH_LINK_UART

#include "driver/uart.h"

#define PORT UART_NUM_0

void link_init(void)
{
    ESP_ERROR_CHECK(uart_driver_install(PORT, RX_BUFFER, TX_BUFFER, 0, NULL, 0));
    ESP_ERROR_CHECK(uart_set_baudrate(PORT, CONFIG_BENCH_UART_BAUD));
}

void link_read(void *buf, size_t len)
{
    uint8_t *p = buf;
    while (len > 0) {
        int n = uart_read_bytes(PORT, p, len, portMAX_DELAY);
        if (n > 0) {
            p += n;
            len -= (size_t)n;
        }
    }
}

void link_write(const void *buf, size_t len)
{
    uart_write_bytes(PORT, buf, len);
    uart_wait_tx_done(PORT, portMAX_DELAY);
}

#elif CONFIG_BENCH_LINK_USB_SERIAL_JTAG

#include "driver/usb_serial_jtag.h"

void link_init(void)
{
    usb_serial_jtag_driver_config_t cfg = {
        .tx_buffer_size = TX_BUFFER,
        .rx_buffer_size = RX_BUFFER,
    };
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));
}

void link_read(void *buf, size_t len)
{
    uint8_t *p = buf;
    while (len > 0) {
        int n = usb_serial_jtag_read_bytes(p, len, portMAX_DELAY);
        if (n > 0) {
            p += n;
            len -= (size_t)n;
        }
    }
}

void link_write(const void *buf, size_t len)
{
    usb_serial_jtag_write_bytes(buf, len, portMAX_DELAY);
}

#endif
