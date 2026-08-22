// GoaVoice RAG Frontend Controller

const apiUrlInput = document.getElementById('api-url-input');

// DOM elements
const micBtn = document.getElementById('mic-btn');
const statusText = document.getElementById('status-text');
const outputSection = document.getElementById('output-section');
const queryDisplay = document.getElementById('query-display');
const answerDisplay = document.getElementById('answer-display');
const speakBtn = document.getElementById('speak-btn');
const langSelect = document.getElementById('lang-select');
const textInput = document.getElementById('text-input');
const sendBtn = document.getElementById('send-btn');
const sourcesToggle = document.getElementById('sources-toggle');
const sourcesList = document.getElementById('sources-list');

// Performance Stats Elements
const statEmbed = document.getElementById('stat-embed');
const statFaiss = document.getElementById('stat-faiss');
const statLlm = document.getElementById('stat-llm');
const statTotal = document.getElementById('stat-total');

// State variables
let recognition = null;
let isListening = false;
let currentVoiceResponseText = '';
let currentAnswerLang = 'en';

// Map select values to Speech SDK language codes
const LANG_MAP = {
    'en': 'en-US',
    'hi': 'hi-IN',
    'mr': 'mr-IN',
    'gu': 'gu-IN',
    'ta': 'ta-IN',
    'te': 'te-IN'
};

// Initialize Speech Recognition
function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn('Speech recognition is not natively supported in this browser.');
        return false;
    }
    
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = false;
    
    recognition.onstart = () => {
        isListening = true;
        setMicState('listening', 'Listening... Speak now.');
    };
    
    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        console.log('Recognized speech:', transcript);
        runRAGQuery(transcript);
    };
    
    recognition.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        if (event.error === 'no-speech') {
            setMicState('idle', 'No speech detected. Try again.');
        } else {
            setMicState('idle', `Error: ${event.error}. Use keyboard fallback.`);
        }
        isListening = false;
    };
    
    recognition.onend = () => {
        isListening = false;
        if (micBtn.classList.contains('listening')) {
            setMicState('idle', 'Click the mic to speak');
        }
    };
    
    return true;
}

// Helper to set mic states visually
function setMicState(state, message) {
    micBtn.classList.remove('listening', 'processing', 'speaking');
    statusText.innerText = message;
    
    if (state === 'listening') {
        micBtn.classList.add('listening');
    } else if (state === 'processing') {
        micBtn.classList.add('processing');
    } else if (state === 'speaking') {
        micBtn.classList.add('speaking');
    }
}

// Speech Synthesizer (TTS)
function speakText(text, langCode) {
    if ('speechSynthesis' in window) {
        // Cancel any ongoing speaking
        window.speechSynthesis.cancel();
        
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = LANG_MAP[langCode] || 'en-US';
        
        // Try to match a native voice for the selected language
        const voices = window.speechSynthesis.getVoices();
        const matchingVoice = voices.find(v => v.lang.startsWith(langCode));
        if (matchingVoice) {
            utterance.voice = matchingVoice;
        }
        
        utterance.onstart = () => {
            setMicState('speaking', 'Speaking response...');
        };
        
        utterance.onend = () => {
            setMicState('idle', 'Click the mic to speak');
        };
        
        utterance.onerror = (e) => {
            console.error('TTS error:', e);
            setMicState('idle', 'Click the mic to speak');
        };
        
        window.speechSynthesis.speak(utterance);
    } else {
        console.warn('Text-to-speech is not supported in this browser.');
    }
}

// Trigger RAG backend search and generation API
async function runRAGQuery(queryText) {
    if (!queryText.trim()) return;
    
    // Stop any ongoing speech synthesis
    if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
    }
    
    // Set UI to processing
    setMicState('processing', 'Searching FAISS & generating response...');
    outputSection.classList.add('hidden');
    
    const selectedLang = langSelect.value;
    
    try {
        const apiBaseUrl = apiUrlInput ? apiUrlInput.value.trim() : 'http://127.0.0.1:5000';
        const response = await fetch(`${apiBaseUrl}/api/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                query: queryText,
                lang: selectedLang
            })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status} ${response.statusText}`);
        }
        
        const data = await response.json();
        renderResponse(data, selectedLang);
        
    } catch (error) {
        console.error('Error fetching query response:', error);
        setMicState('idle', 'RAG call failed. Use text input fallback.');
        
        // Show fallback error visualizer card
        queryDisplay.innerText = queryText;
        answerDisplay.innerText = `Error generating grounded response: ${error.message}`;
        outputSection.classList.remove('hidden');
    }
}

// Render dynamic results inside UI
function renderResponse(data, lang) {
    // 1. Text display
    queryDisplay.innerText = data.query;
    answerDisplay.innerText = data.answer;
    currentVoiceResponseText = data.answer;
    currentAnswerLang = lang;
    
    // 2. Latencies
    statEmbed.innerText = `${data.latency.embedding_ms.toFixed(0)}ms`;
    statFaiss.innerText = `${data.latency.search_ms.toFixed(1)}ms`;
    statLlm.innerText = `${data.latency.llm_ms.toFixed(0)}ms`;
    statTotal.innerText = `${data.latency.total_ms.toFixed(0)}ms`;
    
    // 3. Source passages
    sourcesList.innerHTML = '';
    if (data.results && data.results.length > 0) {
        data.results.forEach((res, index) => {
            const srcItem = document.createElement('div');
            srcItem.className = 'source-item';
            
            // Format source score & language details
            const scorePct = (res.score * 100).toFixed(1);
            const sourceLang = res.metadata.target_lang || 'en';
            
            srcItem.innerHTML = `
                <div class="source-meta">
                    <span>Passage ${index + 1} (Score: <strong class="source-score">${scorePct}%</strong>)</span>
                    <span>Lang: ${sourceLang.toUpperCase()} | ID: ${res.chunk_id}</span>
                </div>
                <div class="source-passage">${res.text}</div>
            `;
            sourcesList.appendChild(srcItem);
        });
    } else {
        sourcesList.innerHTML = '<div class="source-passage">No source passages were returned by the FAISS database.</div>';
    }
    
    // Unhide output panel
    outputSection.classList.remove('hidden');
    
    // 4. Voice playback automatically
    speakText(data.answer, lang);
}

// Event Listeners
micBtn.addEventListener('click', () => {
    if (isListening) {
        if (recognition) recognition.stop();
        isListening = false;
        setMicState('idle', 'Click the mic to speak');
    } else {
        const hasSDK = initSpeechRecognition();
        if (hasSDK && recognition) {
            // Cancel any ongoing speaking
            if ('speechSynthesis' in window) {
                window.speechSynthesis.cancel();
            }
            recognition.lang = LANG_MAP[langSelect.value] || 'en-US';
            recognition.start();
        } else {
            setMicState('idle', 'Microphone recognition not supported. Type your query.');
        }
    }
});

// Repeat Speak Button
speakBtn.addEventListener('click', () => {
    if (currentVoiceResponseText) {
        speakText(currentVoiceResponseText, currentAnswerLang);
    }
});

// Text Fallback triggers
sendBtn.addEventListener('click', () => {
    const textVal = textInput.value.trim();
    if (textVal) {
        runRAGQuery(textVal);
        textInput.value = '';
    }
});

textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        const textVal = textInput.value.trim();
        if (textVal) {
            runRAGQuery(textVal);
            textInput.value = '';
        }
    }
});

// Citations toggler
sourcesToggle.addEventListener('click', () => {
    sourcesList.classList.toggle('collapsed');
    const toggleIcon = sourcesToggle.querySelector('.toggle-icon');
    if (sourcesList.classList.contains('collapsed')) {
        toggleIcon.innerText = '▼';
    } else {
        toggleIcon.innerText = '▲';
    }
});

// Warm up SpeechSynthesis voices array (Google Chrome async load)
if ('speechSynthesis' in window) {
    window.speechSynthesis.getVoices();
    window.speechSynthesis.onvoiceschanged = () => {
        window.speechSynthesis.getVoices();
    };
}
