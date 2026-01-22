# Tailscale Funnel Configuration

Connection details for the Drop Ceiling WebSocket server.

---

## Production Server

| Setting | Value |
|---------|-------|
| **Machine Name** | `cvtower` |
| **Tailscale IP** | `100.81.227.53` |
| **Public HTTPS URL** | `https://cvtower.tail830204.ts.net/` |
| **WebSocket URL** | `wss://cvtower.tail830204.ts.net/` |
| **Local Port** | `8765` |

---

## Website Integration

### JavaScript WebSocket Connection

```javascript
const WEBSOCKET_URL = 'wss://cvtower.tail830204.ts.net/';

const socket = new WebSocket(WEBSOCKET_URL);

socket.onopen = () => {
    console.log('Connected to Drop Ceiling server');
};

socket.onmessage = (event) => {
    const data = JSON.parse(event.data);
    // Handle tracking data
};

socket.onclose = () => {
    console.log('Disconnected from server');
};
```

---

## Verification

Test the connection from any machine:

```bash
# Using curl (check if server responds)
curl -I https://cvtower.tail830204.ts.net/

# Using websocat (test WebSocket)
websocat wss://cvtower.tail830204.ts.net/
```

---

## Management

```bash
# Check funnel status
tailscale funnel status

# Disable funnel
sudo tailscale funnel --https=443 off

# Re-enable funnel
sudo tailscale funnel --bg 8765
```

---

*Generated: January 22, 2026*
