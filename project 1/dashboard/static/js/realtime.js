/**
 * SIEM Dashboard - Real-time Updates
 * Server-Sent Events and auto-refresh logic
 */

// SSE connection
let eventSource = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;

// Connect to SSE stream
function connectSSE() {
    if (eventSource) {
        eventSource.close();
    }
    
    try {
        eventSource = new EventSource('/api/stream');
        
        eventSource.onopen = function() {
            console.log('SSE connection established');
            reconnectAttempts = 0;
            updateConnectionStatus(true);
        };
        
        eventSource.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleSSEMessage(data);
            } catch (e) {
                console.error('Error parsing SSE data:', e);
            }
        };
        
        eventSource.onerror = function(error) {
            console.error('SSE error:', error);
            eventSource.close();
            updateConnectionStatus(false);
            
            // Attempt to reconnect
            if (reconnectAttempts < MAX_RECONNECT_ATTEMPTS) {
                reconnectAttempts++;
                const delay = Math.min(1000 * Math.pow(2, reconnectAttempts), 30000);
                console.log(`Reconnecting in ${delay}ms (attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`);
                setTimeout(connectSSE, delay);
            } else {
                showNotification('Lost connection to server. Please refresh the page.', 'danger');
            }
        };
    } catch (error) {
        console.error('Failed to connect SSE:', error);
        updateConnectionStatus(false);
    }
}

// Handle SSE messages
function handleSSEMessage(data) {
    if (data.status === 'connected') {
        console.log('SSE stream connected');
        return;
    }
    
    if (data.error) {
        console.error('SSE error:', data.error);
        return;
    }
    
    // Handle new alert
    if (data.alert_id) {
        handleNewAlert(data);
    }
}

// Handle new alert notification
function handleNewAlert(alert) {
    // Show notification for high severity
    if (alert.severity === 'high') {
        showNotification(`New high-severity alert: ${alert.filename}`, 'danger');
        
        // Play sound (if enabled)
        playAlertSound();
    }
    
    // Update page-specific content
    if (typeof updateAlertFeed === 'function') {
        updateAlertFeed(alert);
    }
    
    // Update counters
    if (typeof updateDashboardCounters === 'function') {
        updateDashboardCounters();
    }
}

// Update connection status indicator
function updateConnectionStatus(isConnected) {
    const indicator = document.getElementById('live-indicator');
    if (indicator) {
        if (isConnected) {
            indicator.className = 'badge bg-success me-3';
            indicator.innerHTML = '<i class="fas fa-circle fa-xs me-1"></i>Live';
        } else {
            indicator.className = 'badge bg-danger me-3';
            indicator.innerHTML = '<i class="fas fa-circle fa-xs me-1"></i>Disconnected';
        }
    }
}

// Play alert sound
function playAlertSound() {
    // Create audio context (if not muted by user)
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.frequency.value = 800;
        oscillator.type = 'sine';
        
        gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.5);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.5);
    } catch (error) {
        console.log('Could not play alert sound:', error);
    }
}

// Polling fallback (if SSE not available)
let pollingInterval = null;

function startPolling(interval = 10000) {
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }
    
    pollingInterval = setInterval(() => {
        fetchLatestData();
    }, interval);
    
    console.log(`Polling started (interval: ${interval}ms)`);
}

function stopPolling() {
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
        console.log('Polling stopped');
    }
}

async function fetchLatestData() {
    try {
        const response = await fetchWithTimeout('/api/statistics');
        if (response.ok) {
            const data = await response.json();
            updateDashboardData(data);
        }
    } catch (error) {
        console.error('Error fetching latest data:', error);
    }
}

function updateDashboardData(data) {
    // Update KPI cards if present
    const elements = {
        'kpi-total-alerts': data.total_alerts,
        'kpi-high-severity': data.high_severity,
        'kpi-medium-severity': data.medium_severity,
        'kpi-outliers': data.total_outliers
    };
    
    for (const [id, value] of Object.entries(elements)) {
        const element = document.getElementById(id);
        if (element && value !== undefined) {
            animateValue(element, parseInt(element.textContent) || 0, value, 500);
        }
    }
}

// Animate number changes
function animateValue(element, start, end, duration) {
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            current = end;
            clearInterval(timer);
        }
        element.textContent = Math.round(current);
    }, 16);
}

// Page visibility API - pause updates when tab not visible
document.addEventListener('visibilitychange', function() {
    if (document.hidden) {
        console.log('Page hidden - pausing updates');
        if (eventSource) {
            eventSource.close();
        }
        stopPolling();
    } else {
        console.log('Page visible - resuming updates');
        connectSSE();
        // Refresh data immediately
        if (typeof fetchLatestData === 'function') {
            fetchLatestData();
        }
    }
});

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Check if browser supports SSE
    if (typeof EventSource !== 'undefined') {
        console.log('SSE supported - connecting...');
        connectSSE();
    } else {
        console.log('SSE not supported - using polling fallback');
        startPolling(10000);
    }
    
    // Cleanup on page unload
    window.addEventListener('beforeunload', function() {
        if (eventSource) {
            eventSource.close();
        }
        stopPolling();
    });
});

console.log('Real-time updates module loaded');

