use super::{BinaryDescriptors, MatchingError};
use opencv::{core::no_array, prelude::DescriptorMatcherTraitConst};

pub use opencv::core::{DMatch, Vector};

pub trait Matcher {
    fn match_descriptors(
        &self,
        desc1: &BinaryDescriptors,
        desc2: &BinaryDescriptors,
    ) -> Result<Vector<Vector<DMatch>>, MatchingError>;
}

pub struct BruteForceHammingMatcher {
    matcher: opencv::features2d::BFMatcher,
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
    use super::*; // Import the parent module's items, including filter_good_matches.

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
}
