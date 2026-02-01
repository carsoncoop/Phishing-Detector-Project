chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.type === "PREDICT_EMAIL") {
    fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: request.text })
    })
      .then(res => res.json())
      .then(data => sendResponse({ prediction: data.prediction }))
      .catch(err => {
        console.error(err);
        sendResponse({ error: "Failed to contact Flask server" });
      });

    // REQUIRED for async response
    return true;
  }
});
