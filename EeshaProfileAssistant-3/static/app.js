// CV Agent Frontend JavaScript
class CVAgentApp {
    constructor() {
        this.chatMessages = document.getElementById('chat-messages');
        this.questionInput = document.getElementById('question-input');
        this.sendButton = document.getElementById('send-button');
        this.loadingModal = new bootstrap.Modal(document.getElementById('loadingModal'));
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Send button click
        this.sendButton.addEventListener('click', () => this.sendQuestion());
        
        // Enter key press
        this.questionInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.sendQuestion();
            }
        });

        // Sample question buttons
        document.querySelectorAll('.sample-question').forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                const question = e.currentTarget.getAttribute('data-question');
                if (question) {
                    this.questionInput.value = question;
                    this.sendQuestion();
                }
            });
        });
    }

    async sendQuestion() {
        const question = this.questionInput.value.trim();
        
        if (!question) {
            this.showError('Please enter a question');
            return;
        }

        // Disable input and show loading
        this.setLoading(true);
        this.addMessage(question, 'user');
        this.questionInput.value = '';

        try {
            // Show typing indicator
            this.showTypingIndicator();

            // Send request to backend
            const response = await fetch('/ask', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ question }),
            });

            // Remove typing indicator
            this.removeTypingIndicator();

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (data.success) {
                this.addMessage(data.response, 'agent');
            } else {
                this.showError(data.error || 'An error occurred');
            }

        } catch (error) {
            this.removeTypingIndicator();
            console.error('Error sending question:', error);
            this.showError('Sorry, I encountered an error. Please try again.');
        } finally {
            this.setLoading(false);
        }
    }

    addMessage(content, sender) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        // Create avatar
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        
        if (sender === 'user') {
            avatarDiv.innerHTML = '<i class="fas fa-user"></i>';
        } else {
            avatarDiv.innerHTML = '<i class="fas fa-sparkles"></i>';
        }
        
        // Create content
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        
        if (sender === 'user') {
            messageText.textContent = content;
        } else {
            messageText.innerHTML = this.formatAgentResponse(content);
        }
        
        messageContent.appendChild(messageText);
        messageDiv.appendChild(avatarDiv);
        messageDiv.appendChild(messageContent);
        this.chatMessages.appendChild(messageDiv);
        this.scrollToBottom();
        
        // Hide suggestions after first message
        this.hideSuggestions();
    }

    hideSuggestions() {
        const suggestionsContainer = document.querySelector('.suggestions-container');
        if (suggestionsContainer) {
            suggestionsContainer.style.display = 'none';
        }
    }

    formatAgentResponse(content) {
        // Convert markdown-like formatting to HTML
        let formatted = content
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/`(.*?)`/g, '<code>$1</code>')
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>');
        
        // Wrap in paragraphs
        formatted = '<p>' + formatted + '</p>';
        
        // Clean up empty paragraphs
        formatted = formatted.replace(/<p><\/p>/g, '');
        
        return formatted;
    }



    showTypingIndicator() {
        const typingDiv = document.createElement('div');
        typingDiv.className = 'message agent-message typing-message';
        typingDiv.id = 'typing-indicator';
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = '<i class="fas fa-sparkles"></i>';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.innerHTML = `
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        
        messageContent.appendChild(messageText);
        typingDiv.appendChild(avatarDiv);
        typingDiv.appendChild(messageContent);
        this.chatMessages.appendChild(typingDiv);
        this.scrollToBottom();
    }

    removeTypingIndicator() {
        const typingIndicator = document.getElementById('typing-indicator');
        if (typingIndicator) {
            typingIndicator.remove();
        }
    }

    showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'message agent-message error-message';
        
        const avatarDiv = document.createElement('div');
        avatarDiv.className = 'message-avatar';
        avatarDiv.innerHTML = '<i class="fas fa-exclamation-triangle"></i>';
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        const messageText = document.createElement('div');
        messageText.className = 'message-text';
        messageText.innerHTML = `<strong>Error:</strong> ${this.escapeHtml(message)}`;
        
        messageContent.appendChild(messageText);
        errorDiv.appendChild(avatarDiv);
        errorDiv.appendChild(messageContent);
        this.chatMessages.appendChild(errorDiv);
        this.scrollToBottom();
    }

    setLoading(isLoading) {
        this.sendButton.disabled = isLoading;
        this.questionInput.disabled = isLoading;
        
        if (isLoading) {
            this.sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
            this.loadingModal.show();
        } else {
            this.sendButton.innerHTML = '<i class="fas fa-arrow-up"></i>';
            this.loadingModal.hide();
        }
    }

    scrollToBottom() {
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }

    escapeHtml(text) {
        const map = {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
        };
        return text.replace(/[&<>"']/g, (m) => map[m]);
    }
}

// Initialize the app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new CVAgentApp();
});

// Add some utility functions for better UX
window.addEventListener('beforeunload', (e) => {
    // Warn user if they're about to leave during a conversation
    const messages = document.querySelectorAll('.message');
    if (messages.length > 1) {
        e.preventDefault();
        e.returnValue = '';
    }
});

// Add keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to send message
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        document.getElementById('send-button').click();
    }
    
    // Escape to focus on input
    if (e.key === 'Escape') {
        document.getElementById('question-input').focus();
    }
});

// Add auto-resize for text input
document.getElementById('question-input').addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 150) + 'px';
});

// Add success/error toast notifications
function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast align-items-center text-white bg-${type} border-0`;
    toast.setAttribute('role', 'alert');
    toast.setAttribute('aria-live', 'assertive');
    toast.setAttribute('aria-atomic', 'true');
    
    toast.innerHTML = `
        <div class="d-flex">
            <div class="toast-body">
                ${message}
            </div>
            <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
        </div>
    `;
    
    // Add to page
    let toastContainer = document.querySelector('.toast-container');
    if (!toastContainer) {
        toastContainer = document.createElement('div');
        toastContainer.className = 'toast-container position-fixed top-0 end-0 p-3';
        document.body.appendChild(toastContainer);
    }
    
    toastContainer.appendChild(toast);
    
    // Show toast
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
    
    // Remove after hiding
    toast.addEventListener('hidden.bs.toast', () => {
        toast.remove();
    });
}

// Add copy functionality for messages
document.addEventListener('click', (e) => {
    if (e.target.classList.contains('copy-message')) {
        const messageContent = e.target.closest('.message').querySelector('.message-content');
        const text = messageContent.textContent;
        
        navigator.clipboard.writeText(text).then(() => {
            showToast('Message copied to clipboard!', 'success');
        }).catch(() => {
            showToast('Failed to copy message', 'danger');
        });
    }
});
