# coding: utf-8
# ══════════════════════════════════════════════════════════════════════════════
# HARD examples — require exact SmartHub product knowledge that the generic
# baseline skill does NOT contain. The LLM cannot guess these from training
# because all specific values are invented product documentation.
#
# Why the baseline scores near ZERO on these:
#   firmware 3.2.1 bug:   baseline gives generic "restart router" advice
#   error E-7734:         baseline doesn't know error codes at all
#   serial prefix SH2-B:  baseline has no serial-based routing logic
#   LED 3 red blinks:     baseline says "note the LED color" without interpretation
#   firmware rollback:    baseline gives generic "contact support" non-answer
#   XR-500 AP mode:       baseline gives generic AP/bridge mode advice (wrong)
#   port 8443 → 8080:     baseline says "check local IP" (misses ports entirely)
#   Gen 2 vs Gen 3 cross-flash: baseline has no device generation awareness
#
# Expected baseline score: ~0.05–0.20
# Expected evolved score:  ~0.70–0.90
# ══════════════════════════════════════════════════════════════════════════════

example_01 = {
    "task_input": (
        "Customer report (verbatim): 'I updated to firmware 3.2.1 last night and now "
        "my WiFi is terribly slow — like dial-up. My Ethernet is completely fine. "
        "I've restarted the device three times.' "
        "What is the root cause and what are the exact steps to fix it?"
    ),
    "expected_behavior": (
        "Root cause: Firmware 3.2.1 introduced a MAC address conflict in the WiFi driver. "
        "This causes WiFi throughput to degrade 50–80% while Ethernet is unaffected — "
        "exactly the symptom described. Restarting does not fix it because the conflict "
        "persists across reboots until the MAC is explicitly reset. "
        "Fix (exact steps): "
        "1) Log into the admin panel at https://<device-ip>:8443 "
        "2) Navigate to Settings → Advanced → Network "
        "3) Click 'Reset MAC to Factory' and confirm "
        "4) Power cycle the SmartHub (unplug 30 seconds, replug) "
        "5) Test WiFi speed after full reboot (~2 minutes) "
        "This resolves the issue in one power cycle. No firmware downgrade is needed "
        "unless the MAC reset does not restore speed — in that case proceed to rollback."
    ),
    "difficulty": "hard",
    "source": "smarthub-fw321-mac-bug",
}

example_02 = {
    "task_input": (
        "A customer's SmartHub admin panel is showing error code E-7734 on the dashboard. "
        "They ask: 'What does E-7734 mean and what should I do?' "
        "The device itself appears to be working (lights normal, LAN clients connected)."
    ),
    "expected_behavior": (
        "Error code E-7734 indicates an upstream ISP DNS resolution failure — "
        "the SmartHub cannot reach the configured DNS servers. This is NOT a fault "
        "with the SmartHub hardware or firmware; it is an ISP-side or configuration issue. "
        "Inform the customer clearly: 'This error means your internet provider's DNS "
        "service is not responding — your SmartHub itself is fine.' "
        "Steps: "
        "1) Ask the customer to flush DNS on their computer "
        "(Windows: ipconfig /flushdns ; macOS: sudo dscacheutil -flushcache). "
        "2) In the SmartHub admin panel, go to Settings → Network → DNS and try "
        "switching to a public DNS: 8.8.8.8 (Google) or 1.1.1.1 (Cloudflare). "
        "3) If the error clears after changing DNS, instruct the customer to contact "
        "their ISP to report DNS instability on their line. "
        "4) If the error persists even with public DNS servers, escalate to Tier 2."
    ),
    "difficulty": "hard",
    "source": "smarthub-error-e7734",
}

example_03 = {
    "task_input": (
        "A customer calls with an intermittent connectivity issue. Before troubleshooting, "
        "they read out their serial number: SH2-B4471928. "
        "How does this serial number change your support approach?"
    ),
    "expected_behavior": (
        "Serial prefix SH2-B identifies a Gen 2 SmartHub from the B-batch production run, "
        "which has a documented bootloader defect. "
        "This batch should be escalated to Tier 2 immediately — do NOT work through "
        "the standard three-step troubleshooting sequence first. "
        "Tier 1 steps (power cycle, factory reset, firmware reflash) will not resolve "
        "issues caused by the bootloader defect and in some cases a factory reset "
        "can make the device unrecoverable from Tier 1. "
        "Tell the customer: 'I can see from your serial number that your device needs "
        "specialist support — I'm going to connect you directly to our Tier 2 team "
        "who have the right tools for your model.' "
        "Document the serial number, reported symptoms, and that escalation was "
        "triggered by SH2-B serial prefix before transferring."
    ),
    "difficulty": "hard",
    "source": "smarthub-serial-sh2b-escalation",
}

example_04 = {
    "task_input": (
        "A customer says their SmartHub is showing 3 rapid red blinks on the status LED, "
        "repeating in a loop. The device still appears to be routing traffic. "
        "What does this LED pattern mean and what should the customer do?"
    ),
    "expected_behavior": (
        "3 rapid red blinks (repeating loop) = overtemperature warning. "
        "The SmartHub's internal temperature has reached or exceeded 85°C. "
        "The device is still routing because it has not yet hit the hard shutdown "
        "threshold, but shutdown is imminent without intervention. "
        "Immediate steps for the customer: "
        "1) Power the device OFF now — do not wait for it to auto-shutdown. "
        "2) Wait 10 minutes for full thermal cooldown before powering back on. "
        "3) Check the rear vents — they must be clear and unobstructed; the device "
        "needs at least 10 cm of clearance on all sides. Remove it from any enclosed "
        "cabinet or shelf pressed against a wall. "
        "4) After repositioning and cooling: power on and monitor. "
        "If 3 red blinks return within 30 minutes of normal operation after clearing "
        "vents, escalate to Tier 2 — this indicates a failing internal component "
        "(thermal paste or fan on Gen 3 devices)."
    ),
    "difficulty": "hard",
    "source": "smarthub-led-3red-overtemp",
}

example_05 = {
    "task_input": (
        "A customer is on firmware 3.2.1, the MAC reset fix did not work, and "
        "WiFi is still slow. They ask how to roll back to the previous firmware version. "
        "Give the exact steps."
    ),
    "expected_behavior": (
        "Firmware rollback exact path: "
        "1) Log into the admin panel at https://<device-ip>:8443 "
        "2) Navigate to Settings → Firmware → Installed Versions "
        "3) The previous firmware (3.1.8) will be listed — click its row to select it "
        "4) Click 'Apply' and confirm the rollback prompt "
        "5) The device will reboot automatically — this takes 3–5 minutes "
        "6) After reboot, confirm the firmware version has reverted to 3.1.8 "
        "   via Settings → About Device "
        "7) Test WiFi speed "
        "Important warnings: "
        "- Only one previous version is stored. If the customer has updated twice since "
        "  3.1.8, that version may no longer be available in Installed Versions. "
        "- Confirm the customer's serial prefix to identify device generation before "
        "  rollback — cross-generation firmware flashing will brick the device."
    ),
    "difficulty": "hard",
    "source": "smarthub-firmware-rollback",
}

example_06 = {
    "task_input": (
        "A customer has a Netgear XR-500 as their main router and wants to add their "
        "SmartHub as a WiFi access point. They configured the SmartHub in AP mode "
        "and set the XR-500 to bridge mode, but clients connecting to the SmartHub "
        "cannot get an IP address. What is wrong and how do you fix it?"
    ),
    "expected_behavior": (
        "The Netgear XR-500 has a known incompatibility with the SmartHub in "
        "AP/bridge mode: the XR-500's bridge implementation does not correctly pass "
        "DHCP requests to a downstream device acting as an access point. "
        "The fix is to NOT use bridge mode on the XR-500. "
        "Correct configuration: "
        "1) Set the SmartHub to AP mode (as they already have). "
        "2) On the XR-500: go to its admin panel → Advanced → DMZ. "
        "3) Enter the SmartHub's WAN IP address as the DMZ host. "
        "4) Leave the XR-500 in standard routing mode (not bridge mode). "
        "With DMZ configured, the XR-500 passes all unmatched traffic to the SmartHub, "
        "which then handles DHCP for its own connected clients correctly. "
        "Clients on the SmartHub will receive IPs from the SmartHub's DHCP pool. "
        "Note: this means clients on the XR-500 and clients on the SmartHub are on "
        "separate subnets — cross-device local discovery (e.g., printers) will not "
        "work without additional static routing configuration."
    ),
    "difficulty": "hard",
    "source": "smarthub-xr500-dmz-ap",
}

example_07 = {
    "task_input": (
        "A customer insists the admin panel is simply not reachable. They have confirmed: "
        "they are on the SmartHub network, tried three different browsers, disabled their "
        "firewall and VPN, and the device's WAN LED is solid green (connected). "
        "The IP they are using is correct. What have you not tried yet?"
    ),
    "expected_behavior": (
        "The missing step is the port. The SmartHub admin panel does not use the "
        "default HTTP port (80) — it runs on port 8443. "
        "The customer needs to use the full URL: https://<device-ip>:8443 "
        "(the browser will likely show a certificate warning on first access — "
        "this is expected and safe to proceed past for local LAN access). "
        "If port 8443 still shows 'connection refused': "
        "Some ISPs push firmware to their own gateway routers that blocks port 8443 "
        "on the local network (rare but documented). Try the fallback: "
        "https://<device-ip>:8080 "
        "Port 8080 is the SmartHub's secondary admin listener and is not blocked by "
        "any known ISP gateway configuration. "
        "If both 8443 and 8080 are unreachable with all the above confirmed, escalate "
        "to Tier 2 — the web interface process has likely crashed and needs recovery."
    ),
    "difficulty": "hard",
    "source": "smarthub-admin-port-8443-8080",
}

example_08 = {
    "task_input": (
        "A customer has two SmartHub devices. Their serial numbers are SH2-A0034571 "
        "and SH3-C0019832. They downloaded firmware version 3.2.1 and want to flash "
        "both devices. Is this safe? What do you warn them about?"
    ),
    "expected_behavior": (
        "This is NOT safe to apply to both devices without checking. "
        "Serial prefix SH2-* = Gen 2 SmartHub. Serial prefix SH3-* = Gen 3 SmartHub. "
        "Firmware 3.2.1 is a Gen 3 firmware release and is ONLY compatible with "
        "SH3-* devices. Flashing Gen 3 firmware onto a Gen 2 device will brick it "
        "permanently — the bootloader is incompatible and there is no recovery path "
        "from Tier 1. "
        "Correct guidance: "
        "- SH3-C0019832 (Gen 3): firmware 3.2.1 can be applied (though note the "
        "  known WiFi MAC bug — the customer may want to wait for 3.2.2). "
        "- SH2-A0034571 (Gen 2): DO NOT flash 3.2.1. The correct firmware channel "
        "  for Gen 2 is the 2.x branch (latest stable: 2.9.4). "
        "  This can be found on the support portal under 'Gen 2 firmware archives'. "
        "Always confirm serial prefix before any firmware operation."
    ),
    "difficulty": "hard",
    "source": "smarthub-gen2-gen3-cross-flash",
}

GOLDEN_EXAMPLES_HARD = [
    example_01, example_02, example_03, example_04,
    example_05, example_06, example_07, example_08,
]
