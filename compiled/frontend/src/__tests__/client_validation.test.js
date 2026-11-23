import { describe, it, expect } from 'vitest';
import { validateScanRequest, HEIGHT_MIN_CM, HEIGHT_MAX_CM } from '../api/client';

describe('API Client Validation', () => {
  const dummyFile = (type = 'image/jpeg', name = 'test.jpg') =>
    new File(['dummy-content'], name, { type });

  it('passes validation when both images and plausible height are provided', () => {
    const front = dummyFile('image/jpeg', 'front.jpg');
    const side = dummyFile('image/png', 'side.png');
    const errors = validateScanRequest({
      frontImage: front,
      sideImage: side,
      heightCm: 175.0,
    });
    expect(errors).toHaveLength(0);
  });

  it('fails when height is missing, non-numeric, or outside valid range', () => {
    const front = dummyFile();
    const side = dummyFile();

    expect(validateScanRequest({ frontImage: front, sideImage: side, heightCm: '' })).toContain(
      'Standing height is required.'
    );

    expect(validateScanRequest({ frontImage: front, sideImage: side, heightCm: 'abc' })).toContain(
      'Standing height is required.'
    );

    const tooShort = validateScanRequest({ frontImage: front, sideImage: side, heightCm: 50.0 });
    expect(tooShort[0]).toMatch(/must be between 100 cm and 250 cm/);

    const tooTall = validateScanRequest({ frontImage: front, sideImage: side, heightCm: 290.0 });
    expect(tooTall[0]).toMatch(/must be between 100 cm and 250 cm/);
  });

  it('fails when front or side images are missing', () => {
    const errors = validateScanRequest({
      frontImage: null,
      sideImage: null,
      heightCm: 180,
    });
    expect(errors).toContain('Front-view image is required.');
    expect(errors).toContain('Side-view image is required.');
  });

  it('rejects unsupported image mime-types', () => {
    const badFront = dummyFile('application/pdf', 'scan.pdf');
    const goodSide = dummyFile('image/webp', 'side.webp');

    const errors = validateScanRequest({
      frontImage: badFront,
      sideImage: goodSide,
      heightCm: 175,
    });
    expect(errors[0]).toMatch(/Front image type 'application\/pdf' is unsupported/);
  });
});
