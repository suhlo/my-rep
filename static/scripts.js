document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("detection-form");
  const resultDiv = document.getElementById("result");

  form.addEventListener("submit", function (event) {
    event.preventDefault();

    const text = document.getElementById("text").value;

    fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    })
      .then((response) => response.json())
      .then((data) => {
        resultDiv.innerHTML = `
                  <b>Detected Language:</b> ${data.language}<br />
                  <b>Naive Bayes Prediction:</b> ${data.nb_prediction}<br />
                  <b>Decision Tree Prediction:</b> ${data.dt_prediction}<br />
                  <b>Naive Bayes Confidence:</b> ${(
                    data.nb_confidence * 100
                  ).toFixed(2)}%<br />
                  <b>Decision Tree Confidence:</b> ${(
                    data.dt_confidence * 100
                  ).toFixed(2)}%<br />
              `;

        // Remove the error class and add the visible class
        resultDiv.classList.remove("error");
        resultDiv.classList.add("visible");
      })
      .catch((error) => {
        console.error("Error:", error);
        resultDiv.textContent = "An error occurred. Please try again.";
        resultDiv.classList.remove("visible");
        resultDiv.classList.add("error");
      });
  });
});
