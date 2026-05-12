/**
 * AI Fine-Tuner Dashboard — Frontend Logic
 *
 * - Live GPU utilization bar animation
 * - Epoch / loss / batch counter updates
 * - Console output with bounded height (no overflow)
 * - "Stop Training" toggles training on/off
 * - "View Inference" opens a modal with a sample prediction
 */

document.addEventListener('DOMContentLoaded', () => {
    const rand = (min, max) => Math.random() * (max - min) + min;

    // ── DOM refs ───────────────────────────────────────────────────────────────
    const gpuFill    = document.querySelector('.gpu-status .fill');
    const gpuText    = document.querySelector('.gpu-text');
    const metrics    = document.querySelectorAll('.metrics-grid .value');
    const lossValue  = metrics[2];          // Validation Loss
    const timeValue  = metrics[3];          // Est. Time Remaining
    const epochValue = metrics[0];          // Epoch counter
    const consoleDiv = document.querySelector('.console');
    const btnStop    = document.getElementById('btn-stop');
    const btnInfer   = document.getElementById('btn-infer');
    const modal      = document.getElementById('inference-modal');
    const modalClose = document.getElementById('modal-close');
    const inferResult= document.getElementById('infer-result');

    // ── State ──────────────────────────────────────────────────────────────────
    let isTraining  = true;
    let currentLoss = 0.842;
    let batch       = 240;
    let epoch       = 14;
    let minutesLeft = 165; // 2h 45m

    const MAX_EPOCH = 20;
    const MAX_BATCH = 1000;
    const CONSOLE_MAX_LINES = 8;

    // ── Console helper ─────────────────────────────────────────────────────────
    const addConsoleLine = (text, cls = '') => {
        const p = document.createElement('p');
        if (cls) p.className = cls;
        p.textContent = text;
        p.style.animation = 'fadeIn 0.3s ease';
        consoleDiv.appendChild(p);
        consoleDiv.scrollTop = consoleDiv.scrollHeight;
        // Remove oldest line when over limit
        while (consoleDiv.children.length > CONSOLE_MAX_LINES) {
            consoleDiv.removeChild(consoleDiv.firstChild);
        }
    };

    // ── Training update loop ───────────────────────────────────────────────────
    const updateMetrics = () => {
        if (!isTraining) return;

        // GPU
        const usage = rand(80, 98);
        gpuFill.style.width = `${usage}%`;
        gpuText.textContent = `${usage.toFixed(1)}% Utilization`;

        // Batch progress
        batch += Math.floor(rand(5, 20));

        if (batch >= MAX_BATCH) {
            batch = 0;
            epoch = Math.min(epoch + 1, MAX_EPOCH);
            epochValue.textContent = `${epoch} / ${MAX_EPOCH}`;
            minutesLeft = Math.max(0, minutesLeft - Math.floor(rand(8, 15)));
            addConsoleLine(`Epoch ${epoch}/${MAX_EPOCH} complete — saving checkpoint…`);

            if (epoch >= MAX_EPOCH) {
                isTraining = false;
                btnStop.textContent = '✓ Training Complete';
                btnStop.disabled = true;
                addConsoleLine('✅ Fine-tuning complete. Model saved to /models/run-14a/final/', 'active');
                return;
            }
        }

        // Loss decreases with noise
        currentLoss = Math.max(0.08, currentLoss - rand(0.0005, 0.003));
        lossValue.innerHTML = `${currentLoss.toFixed(3)} <span class="trend down">↓</span>`;

        // Time remaining
        const h = Math.floor(minutesLeft / 60);
        const m = minutesLeft % 60;
        timeValue.textContent = `${String(h).padStart(2,'0')}h ${String(m).padStart(2,'0')}m`;

        // Console
        if (Math.random() > 0.6) {
            const msgs = [
                `Epoch ${epoch}/${MAX_EPOCH}: Batch ${batch}/${MAX_BATCH} — Loss: ${currentLoss.toFixed(3)}`,
                `LR scheduler step → lr = ${(rand(1.4,2.0)).toFixed(2)}e-5`,
                `Eval: precision@1 = ${rand(87,93).toFixed(1)}%`,
                `Gradient norm: ${rand(0.8,1.4).toFixed(3)}`,
                `Checkpoint saved to /models/run-14a/ckpt-${epoch}/`,
            ];
            addConsoleLine(msgs[Math.floor(rand(0, msgs.length))], Math.random() > 0.8 ? 'active' : '');
        }
    };

    setInterval(updateMetrics, 1200);

    // ── Stop / Resume button ───────────────────────────────────────────────────
    btnStop.addEventListener('click', () => {
        isTraining = !isTraining;
        if (isTraining) {
            btnStop.textContent = 'Stop Training';
            btnStop.className = 'btn-secondary';
            addConsoleLine('▶ Training resumed.');
        } else {
            btnStop.textContent = '▶ Resume Training';
            btnStop.className = 'btn-primary';
            addConsoleLine('⏸ Training paused by user.');
        }
    });

    // ── Inference modal ────────────────────────────────────────────────────────
    const SAMPLE_QA = [
        {
            q: '¿Cuáles son los síntomas del infarto al miocardio?',
            a: 'Los síntomas incluyen dolor opresivo en el pecho, irradiación al brazo izquierdo, sudoración fría, náuseas y dificultad para respirar. Busque atención médica de inmediato.',
        },
        {
            q: '¿Qué es la hemoglobina A1c?',
            a: 'La HbA1c mide el promedio de glucosa en sangre de los últimos 2-3 meses. Valores <5.7% son normales; 5.7-6.4% indican prediabetes; ≥6.5% se considera diabetes.',
        },
        {
            q: '¿Cómo funciona la terapia LoRA?',
            a: 'LoRA (Low-Rank Adaptation) congela los pesos del modelo base e introduce matrices de bajo rango entrenables en capas de atención, reduciendo parámetros entrenables en ~10,000x.',
        },
    ];

    btnInfer.addEventListener('click', () => {
        const sample = SAMPLE_QA[Math.floor(Math.random() * SAMPLE_QA.length)];
        inferResult.innerHTML = `
            <div class="infer-q"><strong>Pregunta:</strong><br>${sample.q}</div>
            <div class="infer-a"><strong>Respuesta del Modelo:</strong><br>${sample.a}</div>
            <div class="infer-meta">
                Tokens generados: ${Math.floor(rand(40,120))} &nbsp;|&nbsp;
                Latencia: ${rand(0.3,1.2).toFixed(2)}s &nbsp;|&nbsp;
                Temperatura: 0.7
            </div>`;
        modal.style.display = 'flex';
    });

    modalClose.addEventListener('click', () => { modal.style.display = 'none'; });
    modal.addEventListener('click', (e) => { if (e.target === modal) modal.style.display = 'none'; });
});
