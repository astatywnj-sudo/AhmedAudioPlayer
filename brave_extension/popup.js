function setStatus(msg, color) {
  const el = document.getElementById('status');
  el.textContent = msg;
  el.style.color = color || '#e89040';
}

function openInPlayer(url) {
  const protocolUrl = "ahmed-player://play?url=" + encodeURIComponent(url);
  chrome.tabs.create({ url: protocolUrl, active: false });
  setStatus("✅ Sent to player!", "#4ade80");
}

document.getElementById('openPlayer').addEventListener('click', () => {
  chrome.tabs.create({ url: "ahmed-player://open", active: false });
  setStatus("✅ Opening player...", "#4ade80");
});

document.getElementById('playCurrent').addEventListener('click', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab && tab.url) {
    openInPlayer(tab.url);
  } else {
    setStatus("❌ No URL found", "#f87171");
  }
});

document.getElementById('playLink').addEventListener('click', async () => {
  try {
    const text = await navigator.clipboard.readText();
    if (text && text.startsWith('http')) {
      openInPlayer(text);
    } else {
      setStatus("❌ Clipboard is empty", "#f87171");
    }
  } catch (e) {
    setStatus("❌ Cannot read clipboard", "#f87171");
  }
});