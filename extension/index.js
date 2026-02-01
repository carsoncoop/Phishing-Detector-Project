document.addEventListener("DOMContentLoaded", () => {
    const button = document.getElementById("myButton");
    const textarea = document.getElementById("messageBox");
    const resultDiv = document.getElementById("result");

    button.addEventListener("click", async () => {
        const text = textarea.value;

        if (!text.trim()) {
            resultDiv.innerText = "Please paste some text first.";
            return;
        }

        chrome.runtime.sendMessage(
            { type: "PREDICT_EMAIL", text },
            (response) => {
                if (response?.error) {
                    resultDiv.innerText = response.error;
                } else {
                    resultDiv.innerText = "Prediction: " + response.prediction;
    }
  }
);

    });
});
