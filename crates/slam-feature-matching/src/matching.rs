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
    #[inline]
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
