// قائمة سياق (right-click) على أي رابط
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "open-in-ahmed",
    title: "🎵 Open in AhmedAudioPlayer",
    contexts: ["link", "video", "audio"]
  });
});

chrome.contextMenus.onClicked.addListener((info) => {
  if (info.menuItemId === "open-in-ahmed") {
    const url = info.linkUrl || info.srcUrl;
    if (url) {
      openInPlayer(url);
    }
  }
});

function openInPlayer(url) {
  const protocolUrl = "ahmed-player://play?url=" + encodeURIComponent(url);
  chrome.tabs.create({ url: protocolUrl, active: false });
}