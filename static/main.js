// Save this as static/main.js
document.addEventListener("DOMContentLoaded", function () {
  // Check if the loading bar element exists
  const loadingBarContainer = document.getElementById("loading-bar-container");
  if (loadingBarContainer) {
    // Set a timeout to start the loading bar after 1 second
    setTimeout(function () {
      loadingBarContainer.querySelector("#loading-bar").style.width = "100%";
    }, 1000); // 1000 milliseconds = 1 second

    // Optional: Hide the loading bar once it's filled (after 2 seconds)
    setTimeout(function () {
      loadingBarContainer.style.opacity = "0";
    }, 2000); // 2000 milliseconds = 2 seconds
  } else {
    console.error("Loading bar element not found.");
  }
});
