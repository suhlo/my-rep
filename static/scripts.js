// Save this as static/scripts.js
document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("detection-form");
  const resultDiv = document.getElementById("result");

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    const formData = new FormData(event.target);
    fetch("/predict", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        const language = detectLanguage(data.text);
        resultDiv.textContent = "Language: " + language;
        resultDiv.classList.remove("hidden");
        resultDiv.classList.add("visible");
      });
  });
});

// Language detection function
function detectLanguage(text) {
  // Add language detection logic here
  // For example, using a library like lang-detector
  const langDetector = new LangDetector();
  const language = langDetector.detect(text);
  return language;
}