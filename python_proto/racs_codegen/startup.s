/* Minimal Cortex-M4 startup for MPS2-AN386 (QEMU) */
.syntax unified
.cpu cortex-m4
.thumb

.section .vectors, "a"
.word   _estack                  /* 0: initial SP */
.word   Reset_Handler            /* 1: reset */
.word   Default_Handler          /* 2: NMI */
.word   Default_Handler          /* 3: HardFault */
.word   Default_Handler          /* 4: MemManage */
.word   Default_Handler          /* 5: BusFault */
.word   Default_Handler          /* 6: UsageFault */
.word   0,0,0,0                  /* 7-10: reserved */
.word   Default_Handler          /* 11: SVCall */
.word   Default_Handler          /* 12: DebugMon */
.word   0                        /* 13: reserved */
.word   Default_Handler          /* 14: PendSV */
.word   Default_Handler          /* 15: SysTick */
.rept   110                      /* 16-125: IRQ0-109 */
.word   Default_Handler
.endr

.section .text
.thumb_func
.global Reset_Handler
Reset_Handler:
    /* Enable FPU (CP10/CP11 full access) */
    ldr r0, =0xE000ED88       /* CPACR */
    ldr r1, [r0]
    orr r1, r1, #(0xF << 20)  /* CP10, CP11: full access */
    str r1, [r0]
    dsb
    isb
    /* Disable lazy stacking (some QEMU M4 models fault on vpush with LSPEN) */
    ldr r0, =0xE000EF34       /* FPCCR */
    ldr r1, [r0]
    bic r1, r1, #(1 << 30)    /* LSPEN = 0 */
    str r1, [r0]
    /* Ensure FPU active: execute a dummy vmov */
    vmov s0, r2
    vmov r2, s0
    ldr r0, =_sbss
    ldr r1, =_ebss
    movs r2, #0
bss_loop:
    cmp r0, r1
    bge bss_done
    str r2, [r0]
    adds r0, #4
    b bss_loop
bss_done:
    ldr r0, =_sdata
    ldr r1, =_edata
    ldr r2, =_sidata
data_loop:
    cmp r0, r1
    bge data_done
    ldr r3, [r2]
    str r3, [r0]
    adds r0, #4
    adds r2, #4
    b data_loop
data_done:
    bl main
hang:
    b hang

.thumb_func
.global Default_Handler
Default_Handler:
    b .

.section .data
_sidata = .;
