document.addEventListener("DOMContentLoaded", function () {
  // Set a timeout to start the loading bar after 1 second
  setTimeout(function () {
    document.getElementById("loading-bar").style.width = "100%";
  }, 1000); // 1000 milliseconds = 1 second

  // Optional: Hide the loading bar once it's filled (after 2 seconds)
  setTimeout(function () {
    document.getElementById("loading-bar-container").style.opacity = "0";
  }, 2000); // 2000 milliseconds = 2 seconds
});
