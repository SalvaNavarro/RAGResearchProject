/**
 * Functions file for chat logic
 */

// load notification
console.log("INFO: functions.js loaded correctly.");

// getting relevant HTML elements
const inputField = document.querySelector("#input");
const sendButton = document.querySelector("#send");
const messageBox = document.querySelector("#messages");
const grid = document.querySelector("#grid");

// main function to append messages to message box
function addMessage(text, from) {
  // create a new div for the new text and set CSS according to sender
  const message = document.createElement("div");
  message.classList.add("message", from);
  if (from === "human_user") message.classList.add("font-bold");

  // set text as content for message div and nest at bottom of message box
  message.textContent = text;
  messageBox.appendChild(message);
  messageBox.scrollTop = messageBox.scrollHeight;
}

// asynchronous sender function that awaits server RAG response
async function sendMessage(message) {
  // fetch opens a connection to the RAG server with methods/headers/message
  const response = await fetch("/rag", {
    // sending to the /rag relative URL
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }), // request payload, JSON format
  });

  // returning the response (answer + ids) sent by the Flask RAG server
  return response.json();
}

// refreshes the grid with sources returned from the RAG
async function updateGrid(ids) {
  // opens a connection to the server with the new ids, a GET request
  const response = await fetch(`/rag/projects?ids=${ids.join(",")}`);

  // get response from the server and refresh grid
  const html = await response.text();
  grid.innerHTML = html;
}

// data exchange handler function
async function processSend() {
  // read text from input box
  const text = inputField.value.trim();
  if (!text) return;

  // display new user message on chat box and clear input after
  addMessage(text, "human_user");
  inputField.value = "";

  // send message to RAG, capture response in data
  const data = await sendMessage(text);

  // append message on chat as a system response adn refresh grid
  addMessage(data.answer, "rag");
  if (data.project_ids && data.project_ids.length > 0) {
    await updateGrid(data.project_ids);
  }
}

export { inputField, sendButton, processSend };
