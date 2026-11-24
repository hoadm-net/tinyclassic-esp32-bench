#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void link_init(void);
void link_read(void *buf, size_t len);
void link_write(const void *buf, size_t len);

#ifdef __cplusplus
}
#endif
