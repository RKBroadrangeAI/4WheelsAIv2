// LangChain + Jev Web App JavaScript

document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initTicketForm();
    initChatForm();
    initSampleButtons();
});

// Tab Navigation
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.dataset.tab;

            // Update buttons
            tabButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            // Update content
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${tabId}-tab`) {
                    content.classList.add('active');
                }
            });
        });
    });
}

// Ticket Classification Form
function initTicketForm() {
    const form = document.getElementById('ticket-form');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const subject = document.getElementById('subject').value;
        const message = document.getElementById('message').value;
        const submitBtn = form.querySelector('button[type="submit"]');
        const btnText = submitBtn.querySelector('.btn-text');
        const btnLoading = submitBtn.querySelector('.btn-loading');

        // Show loading state
        submitBtn.disabled = true;
        btnText.style.display = 'none';
        btnLoading.style.display = 'inline';

        try {
            const response = await fetch('/api/tickets/classify', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ subject, message }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            displayTicketResults(data);
        } catch (error) {
            console.error('Error:', error);
            alert('Error classifying ticket. Please try again.');
        } finally {
            // Reset button state
            submitBtn.disabled = false;
            btnText.style.display = 'inline';
            btnLoading.style.display = 'none';
        }
    });
}

// Display ticket classification results
function displayTicketResults(data) {
    const resultsCard = document.getElementById('ticket-results');
    
    // Department
    const deptBadge = document.getElementById('result-department');
    deptBadge.textContent = data.department.choice.toUpperCase();
    deptBadge.className = `result-value badge badge-${data.department.choice}`;
    document.getElementById('result-dept-conf').textContent = 
        `${(data.department.confidence * 100).toFixed(1)}% confidence`;

    // Urgency
    const urgBadge = document.getElementById('result-urgency');
    urgBadge.textContent = data.urgency.choice.toUpperCase();
    urgBadge.className = `result-value badge badge-${data.urgency.choice}`;
    document.getElementById('result-urg-conf').textContent = 
        `${(data.urgency.confidence * 100).toFixed(1)}% confidence`;

    // Human Review
    const humanBadge = document.getElementById('result-human');
    const needsHuman = data.requires_human.choice === 'yes';
    humanBadge.textContent = needsHuman ? 'YES' : 'NO';
    humanBadge.className = `result-value badge ${needsHuman ? 'badge-warning' : 'badge-success'}`;
    document.getElementById('result-human-conf').textContent = 
        `${(data.requires_human.confidence * 100).toFixed(1)}% confidence`;

    // Processing Time
    document.getElementById('result-time').textContent = 
        `${data.processing_time_ms.toFixed(2)}ms`;

    // Suggested Response
    document.getElementById('result-response').textContent = 
        data.suggested_response || 'No response generated';

    // Show results
    resultsCard.style.display = 'block';
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Sample Ticket Buttons
function initSampleButtons() {
    const sampleButtons = document.querySelectorAll('.sample-btn');
    
    sampleButtons.forEach(button => {
        button.addEventListener('click', () => {
            document.getElementById('subject').value = button.dataset.subject;
            document.getElementById('message').value = button.dataset.message;
        });
    });
}

// Chat Form
let conversationId = null;

function initChatForm() {
    const form = document.getElementById('chat-form');
    const input = document.getElementById('chat-input');
    const messagesContainer = document.getElementById('chat-messages');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const message = input.value.trim();
        if (!message) return;

        // Add user message to chat
        addChatMessage(message, 'user');
        input.value = '';

        // Send to API
        try {
            const response = await fetch('/api/chat/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message,
                    conversation_id: conversationId,
                }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Store conversation ID
            if (!conversationId) {
                conversationId = crypto.randomUUID();
            }

            // Add assistant response
            addChatMessage(data.response, 'assistant', {
                route: data.route,
                confidence: data.confidence,
                model: data.model_used
            });
        } catch (error) {
            console.error('Error:', error);
            addChatMessage('Sorry, I encountered an error. Please try again.', 'assistant');
        }
    });
}

// Add message to chat
function addChatMessage(content, type, meta = null) {
    const messagesContainer = document.getElementById('chat-messages');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${type}`;
    
    let html = `<p>${escapeHtml(content)}</p>`;
    
    if (meta) {
        html += `<div class="chat-meta">
            Route: ${meta.route} (${(meta.confidence * 100).toFixed(0)}%) · Model: ${meta.model}
        </div>`;
    }
    
    messageDiv.innerHTML = html;
    messagesContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
