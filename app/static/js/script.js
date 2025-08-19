// Enhanced JavaScript for the prediction form
(function () {
    'use strict'

    // Function to restrict input to numbers only
    function restrictToNumbers(input, allowDecimal = false) {
        input.addEventListener('input', function(e) {
            let value = e.target.value;

            if (allowDecimal) {
                // Allow numbers and one decimal point
                value = value.replace(/[^0-9.]/g, '');
                // Ensure only one decimal point
                const parts = value.split('.');
                if (parts.length > 2) {
                    value = parts[0] + '.' + parts.slice(1).join('');
                }
            } else {
                // Allow only numbers
                value = value.replace(/[^0-9]/g, '');
            }

            e.target.value = value;
        });

        // Prevent pasting non-numeric content
        input.addEventListener('paste', function(e) {
            e.preventDefault();
            let paste = (e.clipboardData || window.clipboardData).getData('text');

            if (allowDecimal) {
                paste = paste.replace(/[^0-9.]/g, '');
                // Ensure only one decimal point
                const parts = paste.split('.');
                if (parts.length > 2) {
                    paste = parts[0] + '.' + parts.slice(1).join('');
                }
            } else {
                paste = paste.replace(/[^0-9]/g, '');
            }

            e.target.value = paste;
            // Trigger input event to validate
            e.target.dispatchEvent(new Event('input', { bubbles: true }));
        });

        // Prevent non-numeric keypress
        input.addEventListener('keypress', function(e) {
            const char = String.fromCharCode(e.which);
            if (allowDecimal) {
                if (!/[0-9.]/.test(char) || (char === '.' && e.target.value.includes('.'))) {
                    e.preventDefault();
                }
            } else {
                if (!/[0-9]/.test(char)) {
                    e.preventDefault();
                }
            }
        });
    }

    // Apply number restrictions when DOM is ready
    document.addEventListener('DOMContentLoaded', function() {
        // Integer fields (no decimals)
        const integerFields = [
            'age', 'systolic_bp', 'diastolic_bp', 'resting_heart_rate',
            'height_cm', 'total_cholesterol', 'hdl', 'ldl',
            'fasting_glucose', 'alcohol_per_week', 'max_heart_rate'
        ];

        // Float fields (allow decimals)
        const floatFields = [
            'weight_kg', 'hba1c', 'physical_activity_hours', 'st_depression'
        ];

        // Apply restrictions to integer fields
        integerFields.forEach(fieldId => {
            const field = document.getElementById(fieldId);
            if (field) {
                restrictToNumbers(field, false);
            }
        });

        // Apply restrictions to float fields
        floatFields.forEach(fieldId => {
            const field = document.getElementById(fieldId);
            if (field) {
                restrictToNumbers(field, true);
            }
        });

        // Auto-calculate BMI when height and weight change
        function calculateBMI() {
            const height = document.getElementById('height_cm');
            const weight = document.getElementById('weight_kg');

            if (height && weight && height.value && weight.value) {
                const heightM = height.value / 100;
                const bmi = weight.value / (heightM * heightM);
                console.log('Calculated BMI:', bmi.toFixed(1));

                // Optional: Show BMI in the UI
                let bmiDisplay = document.getElementById('bmi-display');
                if (!bmiDisplay) {
                    bmiDisplay = document.createElement('small');
                    bmiDisplay.id = 'bmi-display';
                    bmiDisplay.className = 'text-muted';
                    weight.parentNode.appendChild(bmiDisplay);
                }
                bmiDisplay.textContent = `BMI: ${bmi.toFixed(1)}`;
            }
        }

        // Add event listeners for BMI calculation
        const heightField = document.getElementById('height_cm');
        const weightField = document.getElementById('weight_kg');

        if (heightField) heightField.addEventListener('input', calculateBMI);
        if (weightField) weightField.addEventListener('input', calculateBMI);

        // Enhanced form validation
        const forms = document.querySelectorAll('.needs-validation');

        Array.prototype.slice.call(forms).forEach(function (form) {
            form.addEventListener('submit', function (event) {
                // Custom validation for select fields
                let isValid = true;
                const selectFields = form.querySelectorAll('select');

                selectFields.forEach(select => {
                    if (select.hasAttribute('required') || select.closest('.form-group').querySelector('label').textContent.includes('*')) {
                        if (!select.value || select.value === '') {
                            isValid = false;
                            select.classList.add('is-invalid');

                            // Add custom error message
                            let errorDiv = select.parentNode.querySelector('.invalid-feedback');
                            if (!errorDiv) {
                                errorDiv = document.createElement('div');
                                errorDiv.className = 'invalid-feedback';
                                select.parentNode.appendChild(errorDiv);
                            }
                            errorDiv.textContent = 'Please select an option from the dropdown.';
                        } else {
                            select.classList.remove('is-invalid');
                            select.classList.add('is-valid');
                        }
                    }
                });

                if (!form.checkValidity() || !isValid) {
                    event.preventDefault();
                    event.stopPropagation();

                    // Scroll to first invalid field
                    const firstInvalid = form.querySelector('.is-invalid, :invalid');
                    if (firstInvalid) {
                        firstInvalid.scrollIntoView({ behavior: 'smooth', block: 'center' });
                        firstInvalid.focus();
                    }
                } else {
                    // Show progress toast
                    const toast = document.getElementById('progressToast');
                    if (toast) {
                        toast.style.display = 'block';
                        const bsToast = new bootstrap.Toast(toast);
                        bsToast.show();
                    }
                }

                form.classList.add('was-validated');
            }, false);
        });

        // Real-time validation for select fields
        const selectFields = document.querySelectorAll('select');
        selectFields.forEach(select => {
            select.addEventListener('change', function() {
                if (this.value && this.value !== '') {
                    this.classList.remove('is-invalid');
                    this.classList.add('is-valid');
                } else if (this.hasAttribute('required') || this.closest('.form-group').querySelector('label').textContent.includes('*')) {
                    this.classList.add('is-invalid');
                    this.classList.remove('is-valid');
                }
            });
        });

        // Form reset functionality
        const resetButton = document.querySelector('button[type="reset"]');
        if (resetButton) {
            resetButton.addEventListener('click', function() {
                // Reset form validation classes
                const form = document.querySelector('.needs-validation');
                if (form) {
                    form.classList.remove('was-validated');

                    // Remove all validation classes
                    const fields = form.querySelectorAll('.is-valid, .is-invalid');
                    fields.forEach(field => {
                        field.classList.remove('is-valid', 'is-invalid');
                    });

                    // Clear BMI display
                    const bmiDisplay = document.getElementById('bmi-display');
                    if (bmiDisplay) {
                        bmiDisplay.remove();
                    }
                }
            });
        }

        // Add visual feedback for number inputs
        const numberInputs = document.querySelectorAll('input[type="number"], input[id$="_bp"], input[id$="_rate"], input[id$="cholesterol"], input[id$="glucose"]');
        numberInputs.forEach(input => {
            input.addEventListener('focus', function() {
                this.setAttribute('placeholder', 'Enter numbers only');
            });

            input.addEventListener('blur', function() {
                this.removeAttribute('placeholder');
            });
        });
    });

    // Additional validation messages
    function addCustomValidationMessage(fieldId, message) {
        const field = document.getElementById(fieldId);
        if (field) {
            field.addEventListener('invalid', function() {
                this.setCustomValidity(message);
            });

            field.addEventListener('input', function() {
                this.setCustomValidity('');
            });
        }
    }

    // Set custom validation messages
    document.addEventListener('DOMContentLoaded', function() {
        addCustomValidationMessage('age', 'Please enter a valid age between 18 and 120 years.');
        addCustomValidationMessage('systolic_bp', 'Please enter a valid systolic blood pressure between 80 and 220 mmHg.');
        addCustomValidationMessage('diastolic_bp', 'Please enter a valid diastolic blood pressure between 40 and 120 mmHg.');
        // Add more custom messages as needed
    });

})();