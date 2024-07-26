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
        resultDiv.textContent = "Language: " + data.language;
        resultDiv.classList.remove("hidden");
        resultDiv.classList.add("visible");
      });
  });
});
