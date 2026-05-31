# coding: utf-8
# Medium examples — RTOS-specific issues where baseline gives irrelevant advice
GOLDEN_EXAMPLES_MEDIUM = [
    # ── 1. malloc() in ISR ──────────────────────────────────────────────────
    {
        "task_input": (
            "Review this FreeRTOS interrupt handler:\n"
            "```c\n"
            "void DMA1_Stream0_IRQHandler(void) {\n"
            "    if (DMA1->LISR & DMA_LISR_TCIF0) {\n"
            "        DMA1->LIFCR = DMA_LIFCR_CTCIF0;\n"
            "        uint8_t *pkt = malloc(DMA_PACKET_SIZE);\n"
            "        if (pkt) {\n"
            "            memcpy(pkt, dma_rx_buf, DMA_PACKET_SIZE);\n"
            "            enqueue_packet(pkt);\n"
            "        }\n"
            "    }\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "Critical: `malloc()` called inside ISR `DMA1_Stream0_IRQHandler`. "
            "The C heap functions use an internal `__malloc_lock()` mutex that can be held "
            "by a task — calling `malloc()` from interrupt context causes a `deadlock` or "
            "`HardFault`. "
            "Fix: pre-allocate a static packet pool or use FreeRTOS `pvPortMalloc()` only "
            "from task context. In the ISR, write directly into a `ring_buffer` protected by "
            "`taskENTER_CRITICAL_FROM_ISR()` / `taskEXIT_CRITICAL_FROM_ISR()`, "
            "then notify the processing task with `xTaskNotifyFromISR()`."
        ),
        "difficulty": "medium",
        "source": "rtos-isr-malloc",
    },
    # ── 2. volatile missing on MMIO-shared variable ─────────────────────────
    {
        "task_input": (
            "Review this ISR and task sharing a counter:\n"
            "```c\n"
            "uint32_t tick_count = 0;\n"
            "\n"
            "void SysTick_Handler(void) {\n"
            "    tick_count++;\n"
            "}\n"
            "\n"
            "void delay_ms(uint32_t ms) {\n"
            "    uint32_t start = tick_count;\n"
            "    while ((tick_count - start) < ms) { /* wait */ }\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "`tick_count` is shared between ISR `SysTick_Handler` and task code `delay_ms` "
            "but is not declared `volatile`. The compiler may cache `tick_count` in a "
            "register inside the `while` loop and never re-read it from memory, causing an "
            "infinite loop. "
            "Fix: `volatile uint32_t tick_count = 0;`. "
            "Also the 32-bit subtraction `(tick_count - start) < ms` handles wraparound "
            "correctly only if both are `uint32_t` (unsigned); ensure no signed promotion."
        ),
        "difficulty": "medium",
        "source": "rtos-volatile-shared",
    },
    # ── 3. Mutex (blocking) taken inside ISR ────────────────────────────────
    {
        "task_input": (
            "Review this FreeRTOS interrupt handler:\n"
            "```c\n"
            "SemaphoreHandle_t uart_mutex;\n"
            "\n"
            "void USART1_IRQHandler(void) {\n"
            "    xSemaphoreTake(uart_mutex, portMAX_DELAY);\n"
            "    log_char(USART1->DR);\n"
            "    xSemaphoreGive(uart_mutex);\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "Critical: `xSemaphoreTake()` with `portMAX_DELAY` called from ISR "
            "`USART1_IRQHandler` — ISR context cannot block. "
            "FreeRTOS `xSemaphoreTake()` will call `vTaskSuspend()` internally when the "
            "semaphore is unavailable, which is illegal from interrupt context and triggers "
            "`configASSERT` or a `HardFault`. "
            "Fix: use `xSemaphoreTakeFromISR(uart_mutex, &xHigherPriorityTaskWoken)` (returns "
            "immediately with `pdFALSE` if unavailable) and call "
            "`portYIELD_FROM_ISR(xHigherPriorityTaskWoken)` at the end of the ISR."
        ),
        "difficulty": "medium",
        "source": "rtos-isr-semaphore",
    },
    # ── 4. xQueueSend with portMAX_DELAY in ISR ─────────────────────────────
    {
        "task_input": (
            "Review this FreeRTOS UART receive ISR:\n"
            "```c\n"
            "QueueHandle_t rx_queue;\n"
            "\n"
            "void USART2_IRQHandler(void) {\n"
            "    if (USART2->SR & USART_SR_RXNE) {\n"
            "        char c = (char)USART2->DR;\n"
            "        xQueueSend(rx_queue, &c, portMAX_DELAY);\n"
            "    }\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "Critical: `xQueueSend()` with `portMAX_DELAY` called from ISR "
            "`USART2_IRQHandler` will attempt to block if the queue is full — "
            "ISR cannot block, causing `HardFault` or `configASSERT` failure. "
            "Replace with `xQueueSendFromISR(rx_queue, &c, &xHigherPriorityTaskWoken)` "
            "which returns `errQUEUE_FULL` immediately without blocking. "
            "Always end the ISR with `portYIELD_FROM_ISR(xHigherPriorityTaskWoken)` "
            "so a woken higher-priority task runs immediately after the ISR returns."
        ),
        "difficulty": "medium",
        "source": "rtos-isr-queue",
    },
    # ── 5. portYIELD_FROM_ISR missing after Give ────────────────────────────
    {
        "task_input": (
            "Review this FreeRTOS GPIO ISR:\n"
            "```c\n"
            "SemaphoreHandle_t button_sem;\n"
            "\n"
            "void EXTI0_IRQHandler(void) {\n"
            "    EXTI->PR = EXTI_PR_PR0;\n"
            "    BaseType_t woken = pdFALSE;\n"
            "    xSemaphoreGiveFromISR(button_sem, &woken);\n"
            "    /* end of ISR */\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "Missing `portYIELD_FROM_ISR(woken)` after `xSemaphoreGiveFromISR()`. "
            "Without it, even if `woken == pdTRUE` (a higher-priority task was unblocked), "
            "that task won't preempt until the current running task voluntarily yields or "
            "its time-slice expires — adding up to one full tick of unnecessary latency. "
            "Add `portYIELD_FROM_ISR(woken)` as the last statement in the ISR to trigger "
            "an immediate context switch when a higher-priority task becomes ready."
        ),
        "difficulty": "medium",
        "source": "rtos-yield-from-isr",
    },
    # ── 6. Non-reentrant libc function called from multiple tasks ────────────
    {
        "task_input": (
            "Review this utility function used by multiple FreeRTOS tasks:\n"
            "```c\n"
            "const char *format_timestamp(time_t t) {\n"
            "    return ctime(&t);  /* called from Task A and Task B concurrently */\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "`ctime()` is not `reentrant` — it returns a pointer to a single shared "
            "`static char[]` buffer internal to the C library. When two FreeRTOS tasks "
            "call `format_timestamp()` concurrently the second call overwrites the buffer "
            "before the first task finishes reading it, causing corrupted timestamps. "
            "Fix: use the `reentrant` variant `ctime_r(&t, buf)` with a caller-supplied "
            "buffer, e.g. `char buf[26]; ctime_r(&t, buf);`, or protect the call with a "
            "`xSemaphoreTake()` / `xSemaphoreGive()` mutex."
        ),
        "difficulty": "medium",
        "source": "rtos-reentrant-libc",
    },
    # ── 7. Recursion with implicit stack depth on small embedded stack ───────
    {
        "task_input": (
            "Review this FreeRTOS task that calls a recursive parser:\n"
            "```c\n"
            "/* Task created with: xTaskCreate(parse_task, ..., 256, NULL, 1, NULL) */\n"
            "void parse_task(void *arg) {\n"
            "    while (1) {\n"
            "        wait_for_packet();\n"
            "        int result = parse_json(rx_buf, rx_len, 0);  /* recursive depth = nesting */\n"
            "        handle_result(result);\n"
            "    }\n"
            "}\n"
            "\n"
            "int parse_json(const char *buf, int len, int depth) {\n"
            "    if (depth > 32) return -1;\n"
            "    /* recurse for nested objects */\n"
            "    return parse_json(buf + offset, remaining, depth + 1);\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "Stack overflow risk: `parse_json()` recurses up to depth 32 inside a FreeRTOS "
            "task with only 256-word (1024-byte) stack. Each recursive frame uses at minimum "
            "~32 bytes for arguments, locals, and return address; 32 frames = ~1 KB, "
            "exhausting the entire task stack. Use `uxTaskGetStackHighWaterMark()` to "
            "measure real usage during testing. Either increase the task stack in "
            "`xTaskCreate()` or rewrite `parse_json()` as an iterative parser with an "
            "explicit stack array to bound memory usage."
        ),
        "difficulty": "medium",
        "source": "rtos-stack-overflow",
    },
    # ── 8. Binary semaphore used for mutual exclusion (no priority inheritance)
    {
        "task_input": (
            "Review this FreeRTOS resource protection code:\n"
            "```c\n"
            "SemaphoreHandle_t spi_lock = xSemaphoreCreateBinary();\n"
            "xSemaphoreGive(spi_lock);  /* initialise as available */\n"
            "\n"
            "void high_prio_task(void *arg) {\n"
            "    xSemaphoreTake(spi_lock, portMAX_DELAY);\n"
            "    spi_transfer(HIGH_DATA);\n"
            "    xSemaphoreGive(spi_lock);\n"
            "}\n"
            "void low_prio_task(void *arg) {\n"
            "    xSemaphoreTake(spi_lock, portMAX_DELAY);\n"
            "    spi_transfer(LOW_DATA);\n"
            "    xSemaphoreGive(spi_lock);\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "`xSemaphoreCreateBinary()` is used for mutual exclusion but binary semaphores "
            "have no `priority_inheritance` protocol. If `low_prio_task` holds `spi_lock` "
            "and a medium-priority task preempts it, `high_prio_task` is blocked waiting for "
            "the lock while the medium task runs — classic `priority_inversion`. "
            "Fix: replace with `xSemaphoreCreateMutex()` which implements FreeRTOS priority "
            "inheritance: the lock holder's priority is temporarily raised to match the "
            "highest waiting task, bounding the inversion window."
        ),
        "difficulty": "medium",
        "source": "rtos-priority-inversion",
    },
]
