# coding: utf-8
# ══════════════════════════════════════════════════════════════════════════════
# EASY examples — general customer support fundamentals the baseline skill covers.
#
# Why the baseline scores well here:
#   - Power cycling, cable checks, ISP verification → all in baseline
#   - Escalation trigger (3 failed attempts) → explicitly in baseline
#   - WiFi SSID/password/range steps → in baseline
#   - Empathetic communication tone → in baseline
#
# Expected baseline score: ~0.65–0.85
# Expected evolved score:  ~0.80–0.95
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "A customer calls to say their SmartHub has no lights at all and the internet "
        "is not working. They have not changed anything recently. "
        "What is the first thing you check and walk them through?"
    ),
    "expected_behavior": (
        "Start with a power cycle: confirm the power cable is firmly plugged into "
        "both the device and the wall socket. Ask if a power strip or surge protector "
        "is involved — try a direct wall connection. Unplug the device, wait 30 seconds, "
        "plug back in. Wait 2 minutes for full boot. If still no lights after that, "
        "try a different power outlet to rule out a dead socket. If there are still "
        "no lights, this may be a hardware fault and should be escalated or flagged "
        "for a replacement assessment. Do not jump straight to factory reset or "
        "firmware troubleshooting on a device that won't power on."
    ),
    "difficulty": "easy",
    "source": "smarthub-power-no-lights",
}

example_02 = {
    "task_input": (
        "A customer cannot connect their laptop to the SmartHub WiFi. "
        "They say they are 'entering the password but it just keeps spinning and fails.' "
        "Walk through the first three troubleshooting steps."
    ),
    "expected_behavior": (
        "Step 1 — Confirm the correct network: Ask the customer to verify they are "
        "selecting the SmartHub's network name (SSID) and not a neighbour's or a "
        "saved network from another location. "
        "Step 2 — Confirm the password: WiFi passwords are case-sensitive. Ask them "
        "to re-enter carefully or show the password field to check for typos. "
        "Step 3 — Forget and reconnect: On the laptop, forget/remove the saved network "
        "entry entirely, then reconnect from scratch. "
        "If all three fail, proceed to check that the SmartHub itself can reach the "
        "internet (test from a device that IS connected, or via Ethernet)."
    ),
    "difficulty": "easy",
    "source": "smarthub-wifi-basic",
}

example_03 = {
    "task_input": (
        "You have now tried three different troubleshooting steps with a customer "
        "and the issue is still not resolved. What do you do next according to policy?"
    ),
    "expected_behavior": (
        "Escalate to Tier 2. Policy requires escalation after three failed troubleshooting "
        "attempts. Before escalating: document the customer's name, device model, serial "
        "number, firmware version if known, and a clear summary of all three steps "
        "attempted and the outcome of each. Provide the customer with a case number "
        "so they can reference the interaction without repeating the story. "
        "Do not promise a resolution timeline. Do not suggest hardware replacement "
        "without Tier 2 confirmation."
    ),
    "difficulty": "easy",
    "source": "smarthub-escalation-policy",
}

example_04 = {
    "task_input": (
        "A customer has been on hold for 25 minutes and opens the conversation "
        "angrily: 'I've been trying to fix this for two days. Your device is garbage.' "
        "How do you respond before any troubleshooting?"
    ),
    "expected_behavior": (
        "Acknowledge the frustration first, before any troubleshooting: "
        "'I completely understand — two days of an issue like this is genuinely "
        "frustrating and I appreciate your patience. I'm here now and focused "
        "entirely on getting this sorted for you.' "
        "Do not be defensive about the device or the company. Do not immediately "
        "launch into troubleshooting questions — take one sentence to show you heard "
        "them. Then transition: 'Can you tell me exactly what you're seeing so I can "
        "get straight to the right fix?' "
        "Use plain language throughout. Avoid unexplained technical jargon. "
        "Confirm each step and summarise the resolution at the end."
    ),
    "difficulty": "easy",
    "source": "smarthub-communication-frustrated",
}

GOLDEN_EXAMPLES_EASY = [example_01, example_02, example_03, example_04]
