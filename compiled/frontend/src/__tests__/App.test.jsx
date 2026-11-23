import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import App from '../App';

describe('Model 1 Testing Frontend App Component', () => {
  beforeEach(() => {
    // Mock URL.createObjectURL and revokeObjectURL in jsdom
    global.URL.createObjectURL = vi.fn(() => 'blob:mock-image-preview');
    global.URL.revokeObjectURL = vi.fn();
  });

  it('renders initial page with height input, dropzones, and scan button', () => {
    render(<App />);

    expect(screen.getByText(/Dual-View Anthropometric Scan/i)).toBeInTheDocument();
    expect(screen.getByTestId('height-input')).toHaveValue(175);
    expect(screen.getByTestId('file-input-front')).toBeInTheDocument();
    expect(screen.getByTestId('file-input-side')).toBeInTheDocument();
    expect(screen.getByTestId('scan-button')).toBeDisabled();
  });

  it('enables scan button when images are selected and height is valid', () => {
    render(<App />);

    const frontFile = new File(['front'], 'front.jpg', { type: 'image/jpeg' });
    const sideFile = new File(['side'], 'side.png', { type: 'image/png' });

    const frontInput = screen.getByTestId('file-input-front');
    const sideInput = screen.getByTestId('file-input-side');

    fireEvent.change(frontInput, { target: { files: [frontFile] } });
    fireEvent.change(sideInput, { target: { files: [sideFile] } });

    const scanButton = screen.getByTestId('scan-button');
    expect(scanButton).not.toBeDisabled();
  });

  it('executes simulated scan and displays results with prominent mock disclaimer', async () => {
    render(<App />);

    const frontFile = new File(['front'], 'front.jpg', { type: 'image/jpeg' });
    const sideFile = new File(['side'], 'side.png', { type: 'image/png' });

    fireEvent.change(screen.getByTestId('file-input-front'), { target: { files: [frontFile] } });
    fireEvent.change(screen.getByTestId('file-input-side'), { target: { files: [sideFile] } });

    const scanButton = screen.getByTestId('scan-button');
    fireEvent.click(scanButton);

    // Verify loading indicator appears
    expect(screen.getByText(/Processing Geometric Slices/i)).toBeInTheDocument();

    // Verify results panel renders
    await waitFor(
      () => {
        expect(screen.getByTestId('results-panel')).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    // Check prominent mock banner
    expect(screen.getByTestId('mock-warning-banner')).toBeInTheDocument();
    expect(screen.getByText(/Simulated Mock Test Data — Not Real Inference/i)).toBeInTheDocument();

    // Check circumferences
    expect(screen.getByTestId('measurement-card-chest')).toBeInTheDocument();
    expect(screen.getByTestId('measurement-card-waist')).toBeInTheDocument();
    expect(screen.getByTestId('measurement-card-abdomen')).toBeInTheDocument();
    expect(screen.getByTestId('measurement-card-hips')).toBeInTheDocument();

    // Check navigation back
    const reScanBtn = screen.getByTestId('re-scan-button');
    fireEvent.click(reScanBtn);
    expect(screen.getByText(/Dual-View Anthropometric Scan/i)).toBeInTheDocument();
  });

  it('displays error banner when simulated failure is toggled', async () => {
    render(<App />);

    const frontFile = new File(['front'], 'front.jpg', { type: 'image/jpeg' });
    const sideFile = new File(['side'], 'side.png', { type: 'image/png' });

    fireEvent.change(screen.getByTestId('file-input-front'), { target: { files: [frontFile] } });
    fireEvent.change(screen.getByTestId('file-input-side'), { target: { files: [sideFile] } });

    // Enable simulated failure checkbox
    const failureCheckbox = screen.getByTestId('simulate-failure-checkbox');
    fireEvent.click(failureCheckbox);

    // Click scan
    fireEvent.click(screen.getByTestId('scan-button'));

    await waitFor(
      () => {
        expect(screen.getByTestId('error-banner')).toBeInTheDocument();
      },
      { timeout: 3000 }
    );

    expect(screen.getByText(/Simulated calibration failure/i)).toBeInTheDocument();
  });
});
