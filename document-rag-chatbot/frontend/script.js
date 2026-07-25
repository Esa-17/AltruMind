// ==============================
// Elements
// ==============================
const hero = document.querySelector(".hero");
const chatArea = document.getElementById("chatArea");
const input = document.getElementById("messageInput");
input.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = input.scrollHeight + "px";
});
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const uploadBtn = document.getElementById("uploadBtn");
const fileInput = document.getElementById("fileInput");
const attachmentPreview =
    document.getElementById("attachmentPreview");
const attachmentName =
    document.getElementById("attachmentName");
const removeAttachment =
    document.getElementById("removeAttachment");
const followups = document.getElementById("followups");
const chatList = document.getElementById("chatList");
const chatHistory = document.getElementById("chatList");
// ==============================
// Session
// ==============================
let sessionId = null;
let firstMessage = true;
let currentChat = null;
let lastQuestion = "";
let messageIndex = 0;
let conversations = {};
let pendingFiles = [];

async function createSession(){
    const res = await fetch("/session/new",{
        method:"POST"
    });
    const data = await res.json();
    sessionId = data.session_id;
    firstMessage = true;
    chatArea.innerHTML = "";
    followups.innerHTML = "";
    hero.style.display = "flex";
    console.log("Session:",sessionId);
}
createSession();
newChatBtn.onclick = async () => {
    chatArea.innerHTML = "";
    followups.innerHTML = "";
    hero.style.display = "flex";
    await createSession();
};
// ==============================
// Add Message
// ==============================
function renderMessage(text, sender){
    const msg = document.createElement("div");
    msg.className = "message " + sender;
    if(sender === "user"){
        msg.innerHTML = `
            <div class="bubble">
                ${text}
            </div>
        `;
    }
    else{
        msg.dataset.raw = text;
        msg.innerHTML = `
            <div class="avatar">
                🧠
            </div>
            <div class="bubble">
                <div class="bot-content">
                    ${marked.parse(text)}
                </div>
                <div class="action-bar">
                    <button class="copy-btn" title="Copy">
                        <i class="fa-regular fa-copy"></i>
                    </button>
                    <button class="regen-btn" title="Regenerate">
                        <i class="fa-solid fa-rotate"></i>
                    </button>
                    <button class="like-btn" title="Good response">
                        <i class="fa-regular fa-thumbs-up"></i>
                    </button>
                    <button class="dislike-btn" title="Bad response">
                        <i class="fa-regular fa-thumbs-down"></i>
                    </button>
                </div>
            </div>
        `;
        attachBotButtonEvents(msg);
    }
    chatArea.appendChild(msg);
}
function addMessage(text,sender){
    renderMessage(text,sender);
    chatArea.scrollTop=chatArea.scrollHeight;
    if(!conversations[sessionId]){
        conversations[sessionId]=[];
    }
    conversations[sessionId].push({
        sender,
        text
    });
}

function addChatHistory(title){
    const item = document.createElement("div");
    item.className = "chat-item";
    item.innerText = title;
    chatHistory.prepend(item);
}
function attachBotButtonEvents(bot){
    const copyBtn = bot.querySelector(".copy-btn");
    const regenBtn = bot.querySelector(".regen-btn");
    const likeBtn = bot.querySelector(".like-btn");
    const dislikeBtn = bot.querySelector(".dislike-btn");
    copyBtn.onclick = async () => {
        await navigator.clipboard.writeText(bot.dataset.raw);
        copyBtn.innerHTML =
            '<i class="fa-solid fa-check"></i>';
        setTimeout(()=>{
            copyBtn.innerHTML =
            '<i class="fa-regular fa-copy"></i>';
        },2000);
    };
    regenBtn.onclick = () => {
        input.value = lastQuestion;
        sendMessage();
    };
    likeBtn.onclick = async () => {
        likeBtn.innerHTML =
        '<i class="fa-solid fa-thumbs-up"></i>';
        likeBtn.style.background="#2E7D32";
        likeBtn.style.color="white";
        dislikeBtn.disabled=true;
        await fetch("/feedback",{
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                session_id:sessionId,
                message_idx:messageIndex,
                feedback:"up"
            })
        });
    };
    dislikeBtn.onclick = async () => {
        dislikeBtn.innerHTML =
        '<i class="fa-solid fa-thumbs-down"></i>';
        dislikeBtn.style.background="#C62828";
        dislikeBtn.style.color="white";
        likeBtn.disabled=true;
        await fetch("/feedback",{
            method:"POST",
            headers:{
                "Content-Type":"application/json"
            },
            body:JSON.stringify({
                session_id:sessionId,
                message_idx:messageIndex,
                feedback:"down"
            })
        });
    };
}
// ==============================
// Send Message
// ==============================
async function sendMessage(){
    try{
    const text = input.value.trim();
    lastQuestion = text;
    if(text==="") return;
    hero.style.display="none";
    if(pendingFiles.length > 0){
        const attach = document.createElement("div");
        attach.className = "user-attachment";
        attach.innerHTML = pendingFiles.map(file => `
            <div>
                <i class="fa-solid fa-file-lines"></i>
                <span>${file.name}</span>
            </div>
        `).join("");
        chatArea.appendChild(attach);
    }
    addMessage(text,"user");
    if(firstMessage){
        firstMessage = false;
        const chatSession = sessionId;
        currentChat = document.createElement("div");
        currentChat.className = "chat";
        currentChat.innerText =
            text.length > 30
            ? text.substring(0,30)+"..."
            : text;
        currentChat.onclick = () => {
            sessionId = chatSession;
            hero.style.display = "none";
            chatArea.innerHTML = "";
            followups.innerHTML = "";
            if(conversations[chatSession]){
                conversations[chatSession].forEach(msg=>{
                    renderMessage(msg.text,msg.sender);
                });
            }
        };
        chatHistory.appendChild(currentChat);
    }
    input.value="";
    input.style.height="auto";
    followups.innerHTML="";
    // Create empty bot message
    const bot = document.createElement("div");
    bot.dataset.raw = "";
    bot.className = "message bot";
    bot.innerHTML = `
    <div class="avatar">
        🧠
    </div>
    <div class="bubble thinking">
        <div class="thinking-text">
            🧠AltruMind is thinking
            <span class="dots"></span>
        </div>
    </div>
    `;
    let firstToken = true;
    chatArea.appendChild(bot);
    chatArea.scrollTop = chatArea.scrollHeight;
    chatArea.scrollTop=chatArea.scrollHeight;
    if(pendingFiles.length > 0){
        const filesToUpload = [...pendingFiles];
        pendingFiles = [];
        fileInput.value = "";
        attachmentPreview.style.display = "none";
        for(const file of filesToUpload){
            const form = new FormData();
            form.append("file", file);
            const uploadRes = await fetch(
                `/upload?session_id=${sessionId}`,
                {
                    method:"POST",
                    body:form
                }
            );
            if(!uploadRes.ok){
                const err = await uploadRes.json();
                alert(err.detail);
                return;
            }
        }
    }
    console.log("Sending request...");
    const response=await fetch("/chat/stream",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({
            session_id:sessionId,
            question:text
        })
    });
    if (!response.ok) {
        const err = await response.text();
        console.error(err);
        addMessage("Server Error:\n" + err, "bot");
        return;
    }
    const reader=response.body.getReader();
    const decoder=new TextDecoder();
    while(true){
        const {done,value}=await reader.read();
        if(done) break;
        const chunk=decoder.decode(value);
        console.log(chunk);
        const lines=chunk.split("\n");
        for(const line of lines){
            if(!line.startsWith("data: ")) continue;
            const data=JSON.parse(line.substring(6));
            if (data.type === "replace") {
                bot.dataset.raw = data.text;
            }
            else if (data.type === "token") {
                if(firstToken){
                    firstToken = false;
                    bot.innerHTML = `
                    <div class="avatar">
                        🧠
                    </div>
                    <div class="bubble">
                        <div class="bot-content"></div>
                    </div>
                    `;
                }
                bot.dataset.raw = (bot.dataset.raw || "") + data.text;
                bot.querySelector(".bot-content").innerHTML =
                    marked.parse(bot.dataset.raw || "") +
                    '<span class="typing-cursor">▍</span>';

                    chatArea.scrollTop = chatArea.scrollHeight;
            }
            else if(data.type==="sources"){
                bot.sources = data.sources;
            }
            else if (data.type === "followups") {
                followups.innerHTML = "";
                data.questions.forEach(q => {
                    const btn = document.createElement("button");
                    btn.className = "followup-btn";
                    btn.innerText = q;
                    btn.onclick = () => {
                        input.value = q;
                        sendMessage();
                    };
                    followups.appendChild(btn);
                });
            }
            else if(data.type==="done"){
                const content = marked.parse(bot.dataset.raw || "");
                let sourcesHTML = "";
                if(bot.sources && bot.sources.length){
                sourcesHTML = `
                    <div class="sources">
                        <div class="sources-title">
                            📚 Sources
                        </div>
                        ${bot.sources.map(s => `
                            <div class="source-item">
                                📄 ${s}
                            </div>
                        `).join("")}
                    </div>
                `;
                }
                bot.innerHTML = `
                <div class="avatar">
                    🧠
                </div>
                <div class="bubble">
                    <div class="bot-content">
                        ${content}
                    </div>
                    ${sourcesHTML}
                    <div class="action-bar">
                        <button class="copy-btn" title="Copy">
                            <i class="fa-regular fa-copy"></i>
                        </button>
                        <button class="regen-btn" title="Regenerate">
                            <i class="fa-solid fa-rotate"></i>
                        </button>
                        <button class="like-btn" title="Good response">
                            <i class="fa-regular fa-thumbs-up"></i>
                        </button>
                        <button class="dislike-btn" title="Bad response">
                            <i class="fa-regular fa-thumbs-down"></i>
                        </button>
                    </div>
                </div>
                `;
                // Reattach button events here
                attachBotButtonEvents(bot);
                messageIndex++;
                if(!conversations[sessionId]){
                    conversations[sessionId]=[];
                }
                conversations[sessionId].push({
                    sender:"bot",
                    text:bot.dataset.raw
                });
            }
        }    
    }
}
catch(err){
        console.error("SEND MESSAGE ERROR:", err);
    }
}
uploadBtn.onclick = () => {
    fileInput.click();
};
fileInput.onchange = () => {
    const files = Array.from(fileInput.files);
    if(files.length === 0) return;
    pendingFiles.push(...files);
    attachmentName.innerHTML = pendingFiles
        .map(file => `📄 ${file.name}`)
        .join("<br>");
    attachmentPreview.style.display = "block";
    console.log(fileInput.files);
    console.log(fileInput.files.length);
};
removeAttachment.onclick = () => {
    pendingFiles = [];
    fileInput.value = "";
    attachmentPreview.style.display = "none";
};
// ==============================
// Events
// ==============================
sendBtn.addEventListener("click",sendMessage);
input.addEventListener("keydown",function(e){
    if(e.key==="Enter" && !e.shiftKey){
        e.preventDefault();
        sendMessage();
    }
});