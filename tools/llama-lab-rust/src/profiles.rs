/// llama-server launch profiles (context size / quality / concurrency tuning).
/// These mirror the C++ autopilot presets tuned for the Qwen 27B on a 24GB card.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    Small,
    Balanced,
    Big,
}

impl Profile {
    pub fn label_key(self) -> &'static str {
        match self {
            Profile::Small => "profile_small",
            Profile::Balanced => "profile_balanced",
            Profile::Big => "profile_big",
        }
    }

    pub fn ctx(self) -> i64 {
        match self {
            Profile::Small => 32768,
            Profile::Balanced => 65536,
            Profile::Big => 98304,
        }
    }

    pub fn cache_kv(self) -> &'static str {
        match self {
            Profile::Small => "q8_0",
            Profile::Balanced => "q5_0",
            Profile::Big => "q5_0",
        }
    }

    pub fn n_predict(self) -> i64 {
        match self {
            Profile::Small => 8192,
            Profile::Balanced => 16384,
            Profile::Big => 32768,
        }
    }

    pub fn threads(self) -> i64 {
        24
    }

    pub fn mtp(self) -> bool {
        true
    }
}

pub fn cycle(profile: Profile) -> Profile {
    match profile {
        Profile::Small => Profile::Balanced,
        Profile::Balanced => Profile::Big,
        Profile::Big => Profile::Small,
    }
}