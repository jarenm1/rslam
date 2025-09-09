use super::{BinaryDescriptors, MatchingError};
use opencv::{
    core::{Ptr, no_array},
    prelude::*,
};

pub use opencv::core::{DMatch, Vector};

pub trait Matcher {
    fn match_descriptors(
        &self,
        desc1: &BinaryDescriptors,
        desc2: &BinaryDescriptors,
    ) -> Result<Vector<Vector<DMatch>>, MatchingError>;
}

pub struct BruteForceHammingMatcher {
    matcher: Ptr<opencv::features2d::BFMatcher>,
}

impl Matcher for BruteForceHammingMatcher {
    fn match_descriptors(
        &self,
        desc1: &BinaryDescriptors,
        desc2: &BinaryDescriptors,
    ) -> Result<Vector<Vector<DMatch>>, MatchingError> {
        let mut matches = Vector::new();
        self.matcher.knn_train_match(
            &*desc1.as_mat_view()?,
            &*desc2.as_mat_view()?,
            &mut matches,
            2,
            &no_array(),
            false,
        )?;

        Ok(matches)
    }
}

pub enum MatchingModel {
    BruteForceHamming(BruteForceHammingMatcher),
}

impl MatchingModel {
    pub fn new_brute_force_hamming() -> Result<Self, MatchingError> {
        let matcher = opencv::features2d::BFMatcher::create(opencv::core::NORM_HAMMING, false)?;
        Ok(Self::BruteForceHamming(BruteForceHammingMatcher {
            matcher,
        }))
    }
}

impl Matcher for MatchingModel {
    fn match_descriptors(
        &self,
        desc1: &BinaryDescriptors,
        desc2: &BinaryDescriptors,
    ) -> Result<Vector<Vector<DMatch>>, MatchingError> {
        match self {
            MatchingModel::BruteForceHamming(matcher) => matcher.match_descriptors(desc1, desc2),
        }
    }
}

pub fn filter_matches(matches: &Vector<Vector<DMatch>>, ratio: f32) -> Vector<DMatch> {
    let good_matches: Vec<DMatch> = matches
        .iter()
        .filter_map(|match_pair| {
            if let (Ok(m), Ok(n)) = (match_pair.get(0), match_pair.get(1)) {
                if m.distance < ratio * n.distance {
                    Some(m.clone())
                } else {
                    None
                }
            } else {
                None
            }
        })
        .collect();

    Vector::from_slice(&good_matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_good_matches_with_valid_and_invalid_data() {
        let mut matches_source = Vector::<Vector<DMatch>>::new();

        matches_source.push(Vector::from_slice(&[
            DMatch::new(0, 0, 0.6).unwrap(),
            DMatch::new(0, 0, 0.9).unwrap(),
        ]));

        matches_source.push(Vector::from_slice(&[
            DMatch::new(1, 1, 0.8).unwrap(),
            DMatch::new(1, 1, 0.9).unwrap(),
        ]));

        matches_source.push(Vector::from_slice(&[
            DMatch::new(2, 2, 0.4).unwrap(),
            DMatch::new(2, 2, 0.8).unwrap(),
        ]));

        matches_source.push(Vector::from_slice(&[DMatch::new(3, 3, 0.5).unwrap()]));

        let ratio = 0.75;

        let good_matches = filter_matches(&matches_source, ratio);

        assert_eq!(good_matches.len(), 2);

        assert_eq!(good_matches.get(0).unwrap().distance, 0.6);
        assert_eq!(good_matches.get(1).unwrap().distance, 0.4);
    }

    fn create_test_descriptors(
        num_features: i32,
        descriptor_size: i32,
    ) -> Result<opencv::core::Mat, opencv::Error> {
        use opencv::{
            core::{CV_8UC1, Mat, Size},
            prelude::*,
        };

        let mut mat = unsafe { Mat::new_size(Size::new(descriptor_size, num_features), CV_8UC1)? };

        for i in 0..num_features {
            for j in 0..descriptor_size {
                let value = ((i * descriptor_size + j) % 256) as u8;
                *mat.at_2d_mut::<u8>(i, j)? = value;
            }
        }

        Ok(mat)
    }

    fn create_similar_descriptors(
        base_mat: &opencv::core::Mat,
        noise_level: u8,
    ) -> Result<opencv::core::Mat, opencv::Error> {
        use opencv::prelude::*;

        let mut similar_mat = base_mat.clone();

        // Add small amount of noise to create similar but not identical descriptors
        for i in 0..similar_mat.rows() {
            for j in 0..similar_mat.cols() {
                let original_val = *base_mat.at_2d::<u8>(i, j)?;
                let noise = (i + j) % (noise_level as i32 + 1);
                let new_val = original_val.saturating_add(noise as u8);
                *similar_mat.at_2d_mut::<u8>(i, j)? = new_val;
            }
        }

        Ok(similar_mat)
    }

    #[test]
    fn test_brute_force_matcher_creation() -> Result<(), Box<dyn std::error::Error>> {
        let _matcher = MatchingModel::new_brute_force_hamming()?;
        Ok(())
    }

    #[test]
    fn test_descriptor_matching_basic() -> Result<(), Box<dyn std::error::Error>> {
        let matcher = MatchingModel::new_brute_force_hamming()?;

        let desc1_mat = create_test_descriptors(5, 32)?;
        let desc2_mat = create_similar_descriptors(&desc1_mat, 10)?;

        let desc1 = BinaryDescriptors::from_mat_borrowed(&desc1_mat)?;
        let desc2 = BinaryDescriptors::from_mat_borrowed(&desc2_mat)?;

        let matches = matcher.match_descriptors(&desc1, &desc2)?;

        assert_eq!(matches.len(), 5, "Wrong number of match groups");

        // Each match group should have k=2 matches (we use knn with k=2)
        for i in 0..matches.len() {
            let match_group = matches.get(i)?;
            assert!(match_group.len() > 0, "Empty match group at index {}", i);
        }

        Ok(())
    }

    #[test]
    fn test_identical_descriptors_matching() -> Result<(), Box<dyn std::error::Error>> {
        let matcher = MatchingModel::new_brute_force_hamming()?;

        let desc_mat = create_test_descriptors(3, 32)?;
        let desc1 = BinaryDescriptors::from_mat_borrowed(&desc_mat)?;
        let desc2 = BinaryDescriptors::from_mat_borrowed(&desc_mat)?;

        let matches = matcher.match_descriptors(&desc1, &desc2)?;

        assert_eq!(matches.len(), 3);

        for i in 0..matches.len() {
            let match_group = matches.get(i)?;
            if match_group.len() > 0 {
                let best_match = match_group.get(0)?;
                assert_eq!(
                    best_match.distance, 0.0,
                    "Expected perfect match for identical descriptors"
                );
                assert_eq!(best_match.query_idx, i as i32);
                assert_eq!(best_match.train_idx, i as i32);
            }
        }

        Ok(())
    }

    #[test]
    fn test_no_matches_different_descriptors() -> Result<(), Box<dyn std::error::Error>> {
        let matcher = MatchingModel::new_brute_force_hamming()?;

        let desc1_mat = create_test_descriptors(3, 32)?;
        let mut desc2_mat = create_test_descriptors(3, 32)?;

        for i in 0..desc2_mat.rows() {
            for j in 0..desc2_mat.cols() {
                let original_val = *desc1_mat.at_2d::<u8>(i, j)?;
                *desc2_mat.at_2d_mut::<u8>(i, j)? = !original_val;
            }
        }

        let desc1 = BinaryDescriptors::from_mat_borrowed(&desc1_mat)?;
        let desc2 = BinaryDescriptors::from_mat_borrowed(&desc2_mat)?;

        let matches = matcher.match_descriptors(&desc1, &desc2)?;

        assert_eq!(matches.len(), 3);

        for i in 0..matches.len() {
            let match_group = matches.get(i)?;
            if match_group.len() > 0 {
                let best_match = match_group.get(0)?;
                assert!(
                    best_match.distance > 100.0,
                    "Expected high distance for different descriptors"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_filter_matches_integration() -> Result<(), Box<dyn std::error::Error>> {
        let matcher = MatchingModel::new_brute_force_hamming()?;

        let desc1_mat = create_test_descriptors(4, 32)?;
        let desc2_mat = create_similar_descriptors(&desc1_mat, 5)?;

        let desc1 = BinaryDescriptors::from_mat_borrowed(&desc1_mat)?;
        let desc2 = BinaryDescriptors::from_mat_borrowed(&desc2_mat)?;

        let matches = matcher.match_descriptors(&desc1, &desc2)?;
        let filtered_matches = filter_matches(&matches, 0.7);

        assert!(filtered_matches.len() > 0, "No matches passed the filter");
        assert!(
            filtered_matches.len() <= matches.len(),
            "Filtered matches exceed original count"
        );

        for i in 0..filtered_matches.len() {
            let match_obj = filtered_matches.get(i)?;
            assert!(match_obj.distance >= 0.0, "Negative match distance");
            assert!(match_obj.query_idx >= 0, "Invalid query index");
            assert!(match_obj.train_idx >= 0, "Invalid train index");
        }

        Ok(())
    }

    #[test]
    fn test_single_descriptor_matching() -> Result<(), Box<dyn std::error::Error>> {
        let matcher = MatchingModel::new_brute_force_hamming()?;

        let desc1_mat = create_test_descriptors(1, 32)?;
        let desc2_mat = create_test_descriptors(1, 32)?;

        let desc1 = BinaryDescriptors::from_mat_borrowed(&desc1_mat)?;
        let desc2 = BinaryDescriptors::from_mat_borrowed(&desc2_mat)?;

        let matches = matcher.match_descriptors(&desc1, &desc2)?;

        assert_eq!(matches.len(), 1);

        let match_group = matches.get(0)?;
        assert!(match_group.len() > 0, "No matches for single descriptor");

        Ok(())
    }

    #[test]
    fn test_descriptor_memory_safety() -> Result<(), Box<dyn std::error::Error>> {
        let matcher = MatchingModel::new_brute_force_hamming()?;

        let (desc1_owned, desc2_owned) = {
            let desc1_mat = create_test_descriptors(5, 32)?;
            let desc2_mat = create_similar_descriptors(&desc1_mat, 12)?;

            let desc1 = BinaryDescriptors::from_mat_borrowed(&desc1_mat)?.into_owned();
            let desc2 = BinaryDescriptors::from_mat_borrowed(&desc2_mat)?.into_owned();

            (desc1, desc2)
        };

        let matches = matcher.match_descriptors(&desc1_owned, &desc2_owned)?;
        assert_eq!(matches.len(), 5);

        Ok(())
    }

    #[test]
    fn test_match_distance_properties() -> Result<(), Box<dyn std::error::Error>> {
        let matcher = MatchingModel::new_brute_force_hamming()?;

        let desc1_mat = create_test_descriptors(3, 32)?;
        let desc2_mat = create_similar_descriptors(&desc1_mat, 20)?;

        let desc1 = BinaryDescriptors::from_mat_borrowed(&desc1_mat)?;
        let desc2 = BinaryDescriptors::from_mat_borrowed(&desc2_mat)?;

        let matches = matcher.match_descriptors(&desc1, &desc2)?;

        for i in 0..matches.len() {
            let match_group = matches.get(i)?;
            if match_group.len() >= 2 {
                let first_match = match_group.get(0)?;
                let second_match = match_group.get(1)?;

                assert!(
                    first_match.distance <= second_match.distance,
                    "Matches not sorted by distance: {} > {}",
                    first_match.distance,
                    second_match.distance
                );

                assert!(
                    first_match.distance >= 0.0,
                    "Negative distance in first match"
                );
                assert!(
                    second_match.distance >= 0.0,
                    "Negative distance in second match"
                );
            }
        }

        Ok(())
    }
}
