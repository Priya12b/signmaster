/**
 * SignMaster Global UI & Action Utilities
 * Provides smooth interactive feedback, clipboard operations, duplication, and toasts.
 */
window.SignMaster = {
  showToast: function(message, type = 'success', duration = 3000) {
    let container = document.getElementById('sm-toast-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'sm-toast-container';
      document.body.appendChild(container);
    }

    const toast = document.createElement('div');
    toast.className = `sm-toast sm-toast-${type}`;
    
    let icon = '✓';
    if (type === 'info') icon = 'ℹ';
    if (type === 'warning') icon = '⚠';
    if (type === 'danger') icon = '✕';

    toast.innerHTML = `
      <span style="font-weight: bold; font-size: 1.1rem;">${icon}</span>
      <span style="flex: 1;">${message}</span>
    `;

    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  },

  copyToClipboard: function(text, btnElement = null, successMsg = 'Copied to clipboard!') {
    if (!navigator.clipboard) {
      const textarea = document.createElement('textarea');
      textarea.value = text;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
    } else {
      navigator.clipboard.writeText(text);
    }

    if (btnElement) {
      const origHtml = btnElement.innerHTML;
      btnElement.innerHTML = `<span>✓ Copied!</span>`;
      btnElement.classList.add('btn-success');
      setTimeout(() => {
        btnElement.innerHTML = origHtml;
        btnElement.classList.remove('btn-success');
      }, 1800);
    }

    this.showToast(successMsg, 'success');
  },

  duplicatePlan: function(planId, btnElement = null) {
    if (btnElement) {
      btnElement.disabled = true;
      btnElement.innerHTML = `<span class="spinner-border spinner-border-sm" role="status"></span> Duplicating...`;
    }

    fetch(`/plan/duplicate/${planId}/`, {
      method: 'POST',
      headers: {
        'X-CSRFToken': this.getCSRFToken(),
        'Content-Type': 'application/json'
      }
    })
    .then(res => res.json())
    .then(data => {
      if (data.status === 'success') {
        this.showToast('Study Plan duplicated successfully!', 'success');
        setTimeout(() => {
          if (data.redirect_url) {
            window.location.href = data.redirect_url;
          } else {
            window.location.reload();
          }
        }, 800);
      } else {
        this.showToast(data.message || 'Error duplicating plan', 'danger');
        if (btnElement) {
          btnElement.disabled = false;
          btnElement.innerHTML = `Duplicate Plan`;
        }
      }
    })
    .catch(err => {
      console.error(err);
      this.showToast('Failed to duplicate plan', 'danger');
      if (btnElement) {
        btnElement.disabled = false;
        btnElement.innerHTML = `Duplicate Plan`;
      }
    });
  },

  getCSRFToken: function() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    if (meta) return meta.getAttribute('content');
    const input = document.querySelector('[name=csrfmiddlewaretoken]');
    if (input) return input.value;
    const cookie = document.cookie.split('; ').find(row => row.startsWith('csrftoken='));
    return cookie ? cookie.split('=')[1] : '';
  }
};
