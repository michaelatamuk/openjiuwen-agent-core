# coding: utf-8
# Easy examples — issues any C reviewer should catch (baseline gets partial credit)
GOLDEN_EXAMPLES_EASY = [
    {
        "task_input": (
            "Review this embedded C function:\n"
            "```c\n"
            "void send_packet(const char *payload) {\n"
            "    char buf[64];\n"
            "    strcpy(buf, payload);  /* payload may be longer than 64 bytes */\n"
            "    uart_write(buf, strlen(buf));\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "Buffer overflow: `strcpy()` into fixed 64-byte `buf` without length check. "
            "If `payload` exceeds 63 bytes the stack is corrupted, overwriting the return "
            "address — critical on embedded targets with no MMU. "
            "Replace with `strncpy(buf, payload, sizeof(buf) - 1); buf[sizeof(buf)-1] = '\\0';` "
            "or use `strlcpy()`. Also pass `sizeof(buf)` explicitly to `uart_write()`."
        ),
        "difficulty": "easy",
        "source": "rtos-buffer-safety",
    },
    {
        "task_input": (
            "Review this embedded C snippet:\n"
            "```c\n"
            "uint8_t *frame = malloc(MAX_FRAME_SIZE);\n"
            "memcpy(frame, rx_buffer, rx_len);\n"
            "process_frame(frame);\n"
            "/* frame is never freed */\n"
            "```"
        ),
        "expected_behavior": (
            "Two issues: (1) `malloc()` return is not checked for NULL — on embedded "
            "systems heap is small; a NULL dereference in `memcpy()` causes a HardFault. "
            "Always check `if (frame == NULL) return;` before use. "
            "(2) Memory leak — `frame` is never passed to `free()`. "
            "In a long-running RTOS task this exhausts the heap."
        ),
        "difficulty": "easy",
        "source": "rtos-memory-safety",
    },
    {
        "task_input": (
            "Review this embedded C function:\n"
            "```c\n"
            "int *get_sensor_reading(void) {\n"
            "    int value = read_adc();\n"
            "    return &value;\n"
            "}\n"
            "```"
        ),
        "expected_behavior": (
            "Returns a pointer to a local (stack-allocated) variable `value` — "
            "dangling pointer, undefined behaviour. The stack frame is released on "
            "function return; the caller reads garbage or corrupted data. "
            "Fix: declare `static int value` for a single-threaded context, "
            "or pass an output pointer parameter `void get_sensor_reading(int *out)`, "
            "or allocate on the heap and document ownership."
        ),
        "difficulty": "easy",
        "source": "rtos-stack-lifetime",
    },
    {
        "task_input": (
            "Review this embedded C macro and its usage:\n"
            "```c\n"
            "#define SQUARE(x) x * x\n"
            "\n"
            "int result = SQUARE(a + b);  /* expands to a + b * a + b */\n"
            "```"
        ),
        "expected_behavior": (
            "`SQUARE(x)` macro is missing parentheses around both the argument and the "
            "whole expression. `SQUARE(a + b)` expands to `a + b * a + b` due to operator "
            "precedence, giving the wrong result. "
            "Correct form: `#define SQUARE(x) ((x) * (x))`. "
            "Also note that if `x` has side effects (e.g. `SQUARE(i++)`) it is evaluated "
            "twice; prefer an `inline` function for safety."
        ),
        "difficulty": "easy",
        "source": "rtos-macro-safety",
    },
]
