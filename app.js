document.addEventListener('DOMContentLoaded', () => {
    // Utility
    const random = (min, max) => Math.random() * (max - min) + min;

    // Elements
    const gpuFill = document.querySelector('.gpu-status .fill');
    const gpuText = document.querySelector('.gpu-text');
    const metrics = document.querySelectorAll('.metrics-grid .value');
    const lossValue = metrics[2];
    const timeValue = metrics[3];
    const consoleDiv = document.querySelector('.console');

    let currentLoss = 0.842;
    let batch = 240;

    const updateMetrics = () => {
        // GPU Usage
        const usage = random(82, 98);
        gpuFill.style.width = `${usage}%`;
        gpuText.innerText = `${usage.toFixed(1)}% Utilization`;

        // Training Progress
        batch += Math.floor(random(5, 15));
        if (batch > 1000) batch = 1000;

        // Loss decreases slowly
        currentLoss = currentLoss - random(0.001, 0.005);
        if (currentLoss < 0.1) currentLoss = 0.1;
        
        lossValue.innerHTML = `${currentLoss.toFixed(3)} <span class="trend down">↓</span>`;

        // Update active log line
        const activeLog = consoleDiv.querySelector('.active');
        if (activeLog) {
            activeLog.innerText = `Epoch 14/20: Batch ${batch}/1000 - Loss: ${currentLoss.toFixed(3)}...`;
        }

        // Randomly output new line
        if (Math.random() > 0.85) {
            if(activeLog) activeLog.classList.remove('active');
            const newLog = document.createElement('p');
            newLog.className = 'active';
            
            const events = [
                `Learning rate adjusted to ${(random(1.5, 2.0)).toFixed(2)}e-5`,
                `Saving intermediate weights...`,
                `Eval step: precision@k = ${(random(85, 92)).toFixed(1)}%`,
                `Epoch 14/20: Batch ${batch}/1000 - Loss: ${currentLoss.toFixed(3)}...`
            ];
            
            newLog.innerText = events[Math.floor(random(0, events.length))];
            newLog.style.animation = 'fadeIn 0.3s ease';
            consoleDiv.appendChild(newLog);
            
            // Auto scroll
            consoleDiv.scrollTop = consoleDiv.scrollHeight;

            if (consoleDiv.children.length > 8) {
                consoleDiv.removeChild(consoleDiv.firstChild);
            }
        }
    };

    setInterval(updateMetrics, 1200);
});
