# coding: utf-8
# ══════════════════════════════════════════════════════════════════════════════
# MEDIUM examples — baseline gives partial or directionally wrong answers
# because it lacks SmartHub-specific procedural details.
#
# Why baseline partially fails here:
#   - Factory reset: baseline says "hold until LED changes" (misses 15-sec duration
#     and solid-amber indicator — customer might release too early at 10s)
#   - Admin panel: baseline says "check local IP / try different browser" (misses
#     that the admin panel runs on port 8443 and has a fallback on 8080)
#   - Post-update slowness: baseline gives generic speed tips (misses firmware
#     version check and known 3.2.1 MAC conflict as root cause)
#   - Device auto-shutdown: baseline says "check power supply" (misses
#     overtemperature protection at 85°C and the 10-min cooldown procedure)
#
# Expected baseline score: ~0.30–0.50
# Expected evolved score:  ~0.70–0.85
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "A customer needs to factory reset their SmartHub after nothing else has worked. "
        "They ask: 'How long do I hold the reset button, and how do I know it worked?' "
        "Give them the exact procedure."
    ),
    "expected_behavior": (
        "Hold the recessed reset button on the rear of the device for exactly 15 seconds "
        "using a pin or paperclip. Do not release early. "
        "The LED will turn solid amber when the reset is accepted — that is the signal "
        "to release the button. The device will then reboot automatically, taking "
        "approximately 2–3 minutes. "
        "Warn the customer that all custom settings (WiFi name, password, port forwarding, "
        "static IPs) will be erased and the device returns to factory defaults. "
        "They will need to reconfigure from scratch after reboot. "
        "A common mistake is releasing at 10 seconds — the LED has not yet changed at "
        "that point and the reset will not take effect."
    ),
    "difficulty": "medium",
    "source": "smarthub-factory-reset-duration",
}

example_02 = {
    "task_input": (
        "A customer says they cannot reach the SmartHub admin panel. "
        "They are typing the device IP into the browser but getting a 'connection refused' error. "
        "They have confirmed they are on the SmartHub network and have tried Chrome and Firefox. "
        "What do you try next?"
    ),
    "expected_behavior": (
        "The SmartHub admin panel runs on port 8443 by default — the URL should be "
        "https://<device-ip>:8443 (not just the bare IP, which defaults to port 80). "
        "Ask the customer to enter the full URL with the port explicitly. "
        "If port 8443 still refuses to connect, try port 8080 as a fallback — "
        "some ISPs block outbound port 8443 at the router level, and the SmartHub "
        "also listens on 8080 for this reason. "
        "If both ports fail and the device appears otherwise functional, power cycle "
        "the device and retry. If still inaccessible, escalate to Tier 2 — "
        "this may indicate a failed web interface process requiring a firmware recovery."
    ),
    "difficulty": "medium",
    "source": "smarthub-admin-port",
}

example_03 = {
    "task_input": (
        "A customer reports their SmartHub has been randomly shutting itself off "
        "about 2–3 hours after power-on. No error lights are visible when it shuts down "
        "— it just goes dark. They say the room is 'normal temperature.' "
        "What is the likely cause and what do you tell them to check?"
    ),
    "expected_behavior": (
        "This pattern — working for 2–3 hours then shutting off with no error lights — "
        "is consistent with the SmartHub's built-in overtemperature protection, which "
        "triggers when the internal temperature reaches 85°C and shuts the device down "
        "to prevent damage. "
        "Ask the customer to check the rear vents: if they are covered, blocked by a "
        "wall, pressed against another device, or the device is inside an enclosed "
        "cabinet, airflow is restricted and heat builds up even in a 'normal' room. "
        "Resolution: "
        "1) Ensure at least 10 cm of clearance on all sides, especially the rear. "
        "2) If it has just shut off, wait 10 minutes for the thermal reset before "
        "powering back on — attempting to restart before the cooldown often results "
        "in another immediate shutdown. "
        "3) If the problem persists after improving airflow, escalate to Tier 2 "
        "as a possible failing internal fan (Gen 3 devices) or thermal paste issue."
    ),
    "difficulty": "medium",
    "source": "smarthub-overtemperature-shutdown",
}

example_04 = {
    "task_input": (
        "A customer says WiFi and internet speeds dropped significantly after they "
        "accepted a firmware update notification last night. "
        "Ethernet speeds are fine. Only WiFi is affected. "
        "What is the first thing you check?"
    ),
    "expected_behavior": (
        "Ask for the firmware version that was installed. The customer can find this in "
        "Settings → About Device or on the admin panel dashboard. "
        "Firmware version 3.2.1 has a known WiFi bug: the update introduces a MAC address "
        "conflict in the wireless driver that causes WiFi throughput to degrade by "
        "50–80% while Ethernet remains unaffected — this matches the customer's symptoms "
        "exactly. "
        "If the customer is on 3.2.1: guide them to the fix (Settings → Advanced → "
        "Network → Reset MAC to Factory). After the reset, power cycle the SmartHub. "
        "WiFi speed should return to normal within one boot cycle. "
        "If they are not on 3.2.1, proceed with standard WiFi troubleshooting "
        "(channel congestion, band selection, client driver update)."
    ),
    "difficulty": "medium",
    "source": "smarthub-firmware-speed-drop",
}

GOLDEN_EXAMPLES_MEDIUM = [example_01, example_02, example_03, example_04]
