/**
 * Main JS file for handling RAG front-end.
 */

import { inputField, sendButton, processSend } from "./functions.js";

console.log("INFO: main.js loaded correctly.");

// a. send message on click
sendButton.addEventListener("click", processSend);

// b. send message if Enter key is pressed
inputField.addEventListener("keydown", function (e) {
  if (e.key === "Enter") processSend();
});
