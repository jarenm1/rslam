use super::{BinaryDescriptors, MatchingError};
use opencv::core::Vector;
use opencv::prelude::*;
use opencv::{
    core::{DMatch, no_array},
    prelude::DescriptorMatcherTraitConst,
};

#[derive(Debug, Clone, Copy)]
pub struct Match {
    pub query_idx: i32,
    pub train_idx: i32,
    pub distance: f32,
}

impl From<DMatch> for Match {
    fn from(value: DMatch) -> Self {
        Self {
            query_idx: value.query_idx,
            train_idx: value.train_idx,
            distance: value.distance,
        }
    }
}

pub trait Matcher {
    fn match_descriptors(
        &self,
        desc1: &BinaryDescriptors,
        desc2: &BinaryDescriptors,
    ) -> Result<Vec<Vec<Match>>, MatchingError>;
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
    ) -> Result<Vec<Vec<Match>>, MatchingError> {
        let mut matches = Vector::new();
        self.matcher.knn_train_match(
            &Mat::try_from(desc1)?,
            &Mat::try_from(desc2)?,
            &mut matches,
            2,
            &no_array(),
            false,
        )?;

        let matches_vec: Vec<Vec<Match>> = matches
            .into_iter()
            .map(|inner| {
                inner
                    .into_iter()
                    .map(|dmatch| dmatch.into())
                    .collect::<Vec<Match>>()
            })
            .collect::<Vec<Vec<Match>>>();

        Ok(matches_vec)
    }
}
