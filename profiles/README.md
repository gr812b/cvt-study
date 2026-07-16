# Reusable profile library

This top-level directory documents profile categories a team may maintain outside
individual projects. The package's versioned built-ins are under
`src/cvt_track_study/builtin_profiles/`; a real team's shared profiles may live in
any directory listed by a project's `[profiles].roots`.

Profile IDs must be globally unique. Team profiles should extend built-ins under a
new ID, retain uncertainty and provenance, and override only assumptions the team
has better evidence for.
