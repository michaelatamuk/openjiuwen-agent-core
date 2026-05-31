# coding: utf-8
# Hard examples — baseline gives completely wrong/useless advice; LLM-as-judge ~0.05-0.15
GOLDEN_EXAMPLES_HARD = [
    # ── 1. Non-atomic 64-bit MMIO read torn by ISR ──────────────────────────
    {
        "task_input": (
            "Review this timestamp read function on Cortex-M4:\n"
            "```c\n"
            "typedef struct { uint32_t hi; uint32_t lo; } Ts64;\n"
            "\n"
            "Ts64 read_timer(void) {\n"
            "    Ts64 t;\n"
            "    t.hi = TIMER->CNT_HI;   /* ISR can fire here */\n"
            "    t.lo = TIMER->CNT_LO;\n"
            "    return t;\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "Non-atomic 64-bit read: an ISR can preempt between the `TIMER->CNT_HI` and "
            "`TIMER->CNT_LO` reads. If the counter rolls over in that window, `t.hi` holds "
            "the pre-rollover high word while `t.lo` holds the post-rollover low word — a "
            "torn read producing a wildly incorrect timestamp. "
            "Fix: wrap with `__disable_irq()` / `__enable_irq()` (bare-metal) or "
            "`portENTER_CRITICAL()` / `portEXIT_CRITICAL()` (FreeRTOS). "
            "Alternatively use the hardware 'read-hi, read-lo, re-read-hi, retry if changed' "
            "double-read pattern: read `CNT_HI`, read `CNT_LO`, read `CNT_HI` again; "
            "if the two `hi` values differ, repeat."
        ),
        "difficulty": "hard",
        "source": "rtos-atomic-mmio-read",
    },
    # ── 2. FreeRTOS timer callback doing long blocking work ──────────────────
    {
        "task_input": (
            "Review this FreeRTOS software timer callback:\n"
            "```c\n"
            "void sensor_timer_cb(TimerHandle_t xTimer) {\n"
            "    SensorData d = read_i2c_sensor();   /* ~20 ms blocking I2C */\n"
            "    write_flash_log(d);                  /* ~50 ms flash erase+write */\n"
            "    update_display(d);                   /* ~10 ms SPI */\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "FreeRTOS software timer callbacks execute in the `xTimerTask` daemon task, "
            "which has a single shared stack and processes all timer callbacks serially. "
            "Long-running operations (20 ms I2C + 50 ms flash + 10 ms SPI = 80 ms) block "
            "`xTimerTask`, delaying every other software timer and `xTimerStart()` / "
            "`xTimerStop()` commands queued to the timer command queue. "
            "Fix: in the callback, post a notification with `xTaskNotify()` or "
            "`xQueueSend()` to a dedicated worker task that performs the slow I2C, flash, "
            "and SPI operations. The callback itself should return in microseconds."
        ),
        "difficulty": "hard",
        "source": "rtos-timer-task-blocking",
    },
    # ── 3. volatile missing on memory-mapped peripheral register ────────────
    {
        "task_input": (
            "Review this bare-metal GPIO polling loop:\n"
            "```c\n"
            "#define GPIOA_IDR  (*((uint32_t *)0x40020010))\n"
            "\n"
            "void wait_for_button(void) {\n"
            "    while ((GPIOA_IDR & (1u << 0)) == 0) {\n"
            "        /* wait for PA0 to go high */\n"
            "    }\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "`GPIOA_IDR` is a memory-mapped I/O register but the pointer cast "
            "`(uint32_t *)0x40020010` is not `volatile`. The compiler sees no writes to the "
            "pointed-to location in the loop and is legally allowed to hoist the read out of "
            "the loop (read once, store in register, loop forever). The button press is never "
            "detected. Fix: `#define GPIOA_IDR  (*((volatile uint32_t *)0x40020010))`. "
            "All memory-mapped peripheral registers must be accessed through `volatile` "
            "pointers to prevent compiler optimisation from removing or reordering the access."
        ),
        "difficulty": "hard",
        "source": "rtos-volatile-mmio",
    },
    # ── 4. Missing DSB/DMB memory barrier after DMA setup ───────────────────
    {
        "task_input": (
            "Review this Cortex-M DMA transfer setup:\n"
            "```c\n"
            "uint8_t tx_buf[256];\n"
            "\n"
            "void start_dma_transfer(void) {\n"
            "    fill_buffer(tx_buf, sizeof(tx_buf));\n"
            "    DMA1_Channel1->CMAR  = (uint32_t)tx_buf;\n"
            "    DMA1_Channel1->CNDTR = sizeof(tx_buf);\n"
            "    DMA1_Channel1->CCR  |= DMA_CCR_EN;\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "Missing memory barrier between CPU writes to `tx_buf` and DMA engine reads. "
            "On Cortex-M processors with a write buffer, stores to `tx_buf` from "
            "`fill_buffer()` may still be in the CPU write buffer when the DMA engine begins "
            "reading — the DMA sees stale data. "
            "Insert `__DSB()` (Data Synchronisation Barrier) after `fill_buffer()` and "
            "before enabling the DMA channel: `__DSB(); DMA1_Channel1->CCR |= DMA_CCR_EN;`. "
            "Additionally, `tx_buf` should be declared `__attribute__((aligned(4)))` and, "
            "if the device has a D-cache (Cortex-M7), the cache line must be cleaned with "
            "`SCB_CleanDCache_by_Addr()` before the DMA transfer."
        ),
        "difficulty": "hard",
        "source": "rtos-dma-memory-barrier",
    },
    # ── 5. Stack variable in FreeRTOS task exceeding task stack allocation ───
    {
        "task_input": (
            "Review this FreeRTOS task:\n"
            "```c\n"
            "/* Created with xTaskCreate(compress_task, \"compress\", 512, NULL, 2, NULL) */\n"
            "void compress_task(void *arg) {\n"
            "    uint8_t in_buf[1024];   /* 1 KB */\n"
            "    uint8_t out_buf[1024];  /* 1 KB */\n"
            "    while (1) {\n"
            "        receive_data(in_buf, sizeof(in_buf));\n"
            "        lz4_compress(in_buf, sizeof(in_buf), out_buf, sizeof(out_buf));\n"
            "        transmit_data(out_buf, sizeof(out_buf));\n"
            "    }\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "Stack overflow: task is created with a 512-word (2048-byte on 32-bit ARM) stack "
            "but declares 2 KB of local arrays (`in_buf` + `out_buf`), exhausting the entire "
            "stack before the function body runs. FreeRTOS stack overflow hooks "
            "(`vApplicationStackOverflowHook`) may not fire reliably for this severity. "
            "Fix: increase the stack in `xTaskCreate()` to at least 768 words, or allocate "
            "the buffers as `static` (BSS/data segment) or via `pvPortMalloc()` outside the "
            "loop. Use `uxTaskGetStackHighWaterMark()` to verify the actual high-water mark."
        ),
        "difficulty": "hard",
        "source": "rtos-task-stack-overflow",
    },
    # ── 6. Watchdog not refreshed during long blocking operation ────────────
    {
        "task_input": (
            "Review this FreeRTOS task with an independent watchdog:\n"
            "```c\n"
            "/* IWDG timeout: 500 ms */\n"
            "void data_task(void *arg) {\n"
            "    while (1) {\n"
            "        HAL_IWDG_Refresh(&hiwdg);\n"
            "        xSemaphoreTake(data_ready_sem, portMAX_DELAY);\n"
            "        process_large_dataset();   /* may take up to 2 seconds */\n"
            "    }\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "Watchdog `IWDG` is refreshed (`HAL_IWDG_Refresh()`) before blocking on "
            "`xSemaphoreTake()`, but `process_large_dataset()` can run for up to 2 seconds "
            "— four times the 500 ms `IWDG` timeout. The system will reset mid-processing "
            "with no error indication. "
            "Fix options: (1) refresh the watchdog inside `process_large_dataset()` at "
            "regular checkpoints; (2) move watchdog refresh to a dedicated high-priority "
            "watchdog task that calls `HAL_IWDG_Refresh()` every 250 ms and uses "
            "`xTaskNotify()` / `ulTaskNotifyTake()` as a liveness check from other tasks; "
            "(3) redesign processing to be interruptible with `vTaskDelay()` opportunities."
        ),
        "difficulty": "hard",
        "source": "rtos-watchdog-refresh",
    },
    # ── 7. printf / semihosting in fault handler hangs without debugger ──────
    {
        "task_input": (
            "Review this Cortex-M HardFault handler:\n"
            "```c\n"
            "void HardFault_Handler(void) {\n"
            "    printf(\"HardFault! PC=0x%08lx LR=0x%08lx\\n\",\n"
            "           (unsigned long)fault_pc, (unsigned long)fault_lr);\n"
            "    while (1);\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "`printf()` uses `semihosting` via a BKPT instruction when compiled with "
            "semihosting libraries; if no debugger is attached the CPU halts at `BKPT` "
            "waiting for a debugger that never comes — the fault handler freezes instead of "
            "recovering or logging. Even with a UART-backed `printf()`, the UART driver may "
            "use interrupts or DMA that are unusable from fault context. "
            "Fix: store fault info to `BACKUP_SRAM`, RTC backup registers, or a dedicated "
            "RAM buffer declared `__attribute__((section(\".noinit\")))` to survive reset; "
            "use `ITM_SendChar()` for non-blocking SWO trace output; then call "
            "`NVIC_SystemReset()` to reset cleanly rather than spinning."
        ),
        "difficulty": "hard",
        "source": "rtos-fault-handler-printf",
    },
    # ── 8. FreeRTOS notification used as event flags but bits get ORed wrong
    {
        "task_input": (
            "Review this FreeRTOS event-flag pattern using task notifications:\n"
            "```c\n"
            "#define EVT_SENSOR  (1u << 0)\n"
            "#define EVT_NETWORK (1u << 1)\n"
            "#define EVT_DISPLAY (1u << 2)\n"
            "\n"
            "/* From three different ISRs: */\n"
            "xTaskNotify(worker, EVT_SENSOR,  eSetBits);\n"
            "xTaskNotify(worker, EVT_NETWORK, eSetBits);\n"
            "xTaskNotify(worker, EVT_DISPLAY, eSetBits);\n"
            "\n"
            "/* Worker task: */\n"
            "void worker_task(void *arg) {\n"
            "    while (1) {\n"
            "        uint32_t flags;\n"
            "        xTaskNotifyWait(0, UINT32_MAX, &flags, portMAX_DELAY);\n"
            "        if (flags & EVT_SENSOR)  handle_sensor();\n"
            "        if (flags & EVT_NETWORK) handle_network();\n"
            "        if (flags & EVT_DISPLAY) handle_display();\n"
            "    }\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "The pattern is correct but has a subtle race: `xTaskNotifyWait()` with "
            "`ulBitsToClearOnExit = UINT32_MAX` atomically clears all bits when returning. "
            "If a new `xTaskNotify(..., eSetBits)` arrives *after* `xTaskNotifyWait()` "
            "reads the value but *before* it clears the bits, that new notification bit is "
            "cleared and the event is silently lost. "
            "For ISR callers, use `xTaskNotifyFromISR()` instead of `xTaskNotify()`. "
            "For truly atomic event flag semantics where no events can be lost, use a "
            "FreeRTOS `EventGroupHandle_t` with `xEventGroupSetBitsFromISR()` and "
            "`xEventGroupWaitBits()` — the event group implementation is designed to handle "
            "concurrent setters safely."
        ),
        "difficulty": "hard",
        "source": "rtos-notification-race",
    },
]
