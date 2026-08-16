/* Cycle measurement via SysTick (works on QEMU MPS2-AN386).
   SysTick: 24-bit down counter, CLKSOURCE=1 → processor clock.
   We read remaining count; elapsed = (0xFFFFFF - CVR). */
#include <stdint.h>

#define SYST_CSR   (*(volatile uint32_t*)0xE000E010)
#define SYST_RVR   (*(volatile uint32_t*)0xE000E014)
#define SYST_CVR   (*(volatile uint32_t*)0xE000E018)

void dbg_print_str(const char* s);
void dbg_print_u32(uint32_t v);

static void semihost(uint32_t op, void* arg) {
    register uint32_t r0 asm("r0") = op;
    register void* r1 asm("r1") = arg;
    asm volatile("bkpt 0xAB" : : "r"(r0), "r"(r1) : "memory");
}

void dbg_init(void) {
    SYST_RVR = 0x00FFFFFF;   /* max reload */
    SYST_CVR = 0;
    SYST_CSR = 0x5;          /* CLKSOURCE=1, ENABLE=1 */
}

uint32_t dbg_cycles(void) {
    return (0x00FFFFFF - SYST_CVR) & 0x00FFFFFF;
}

void dbg_print_str(const char* s) {
    semihost(0x04, (void*)s);  /* SYS_WRITE0 */
}

void dbg_print_u32(uint32_t v) {
    char buf[16];
    for (int i = 0; i < 8; i++) {
        buf[7 - i] = "0123456789abcdef"[v & 0xF];
        v >>= 4;
    }
    buf[8] = '\n';
    buf[9] = 0;
    semihost(0x04, (void*)buf);
}
