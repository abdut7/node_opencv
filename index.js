const cv = require('opencv4nodejs');

// Load images
const img1 = cv.imread('./image.png');
const img2 = cv.imread('./p1.png');

// Create SIFT feature detector
const sift = new cv.SIFTDetector();

// Detect key points and compute descriptors for both images
const kp1 = sift.detect(img1);
const desc1 = sift.compute(img1, kp1);
const kp2 = sift.detect(img2);
const desc2 = sift.compute(img2, kp2);

// Create brute-force matcher and match descriptors
const bfMatcher = new cv.BFMatcher(cv.NORM_L2, true);
const matches = bfMatcher.match(desc1, desc2);

// Sort matches by distance
matches.sort((m1, m2) => m1.distance - m2.distance);

// Draw first 100 matches
const numMatches = 100;
const matchedImg = cv.drawMatches(img1, kp1, img2, kp2, matches.slice(0, numMatches));

// Display matched image
cv.imshow('matched image', matchedImg);
cv.waitKey();
