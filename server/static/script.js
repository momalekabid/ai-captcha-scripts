let lastResponseHash = '';

function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        const char = str.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
    }
    return hash;
}

function updateTime() {
    const now = new Date();
    const timeString = now.toLocaleTimeString('en-US', { hour12: false });
    document.getElementById('current-time').textContent = timeString;
    document.getElementById('current-time-2').textContent = timeString;
}

function updateStatus() {
    fetch(window.location.href)
        .then(response => response.text())
        .then(html => {
            const parser = new DOMParser();
            const newDoc = parser.parseFromString(html, 'text/html');
            
            const gameContainer = document.getElementById('game-container');
            const mainContent = document.getElementById('main-content');
            const successMessage = document.getElementById('success-message');
            const failureMessage = document.getElementById('failure-message');
            
            // Update game status
            gameContainer.className = newDoc.getElementById('game-container').className;
            
            if (gameContainer.classList.contains('complete')) {
                successMessage.style.display = 'block';
                failureMessage.style.display = 'none';
                mainContent.style.display = 'none';
            } else if (gameContainer.classList.contains('failed')) {
                successMessage.style.display = 'none';
                failureMessage.style.display = 'block';
                mainContent.style.display = 'none';
            } else {
                successMessage.style.display = 'none';
                failureMessage.style.display = 'none';
                mainContent.style.display = 'flex';
                
                // Update specific elements
                document.getElementById('verification-status').textContent = newDoc.getElementById('verification-status').textContent;
                document.getElementById('current-step').textContent = newDoc.getElementById('current-step').textContent;
                document.getElementById('captcha-display').innerHTML = newDoc.getElementById('captcha-display').innerHTML;
                
                // Update progress bar and text
                const newProgressBar = newDoc.getElementById('progress-bar');
                const progressBar = document.getElementById('progress-bar');
                progressBar.style.width = newProgressBar.style.width;
                
                document.getElementById('progress-text').textContent = newDoc.getElementById('progress-text').textContent;
            }
        });
}

setInterval(updateTime, 1000);
setInterval(updateStatus, 1000);

// Initial calls
updateTime();
updateStatus();