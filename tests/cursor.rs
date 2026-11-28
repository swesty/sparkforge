//! `cursor::Cursor`, the host's record of a runtime's device position.
use sparkforge::cursor::Cursor;

#[test]
fn a_step_that_fails_after_it_began_refuses_reuse_until_set() {
    let mut c = Cursor::new(4);
    c.check_set(3, 9, 10).unwrap();
    c.set(3);
    c.check_step(10, 100).unwrap();
    // The last slot's step launched `argmax_advance`, then its readback failed: the device may be
    // at capacity while the host still says 3.
    c.begin();
    let e = c.check_step(10, 100).unwrap_err().to_string();
    assert!(e.contains("device position is unknown"), "{e}");
    assert_eq!(c.pos(), 3);
    c.set(2);
    c.check_step(10, 100).unwrap();
    // A clean step advances the host, and the next token is the device's greedy one.
    c.begin();
    c.finish_at(3);
    assert!(!c.stale());
    assert!(c.check_step(10, 100).unwrap_err().to_string().contains("embedding rows"), "only 10 of 100 rows");
    c.check_step(100, 100).unwrap();
    c.finish_at(4);
    assert!(c.check_step(100, 100).unwrap_err().to_string().contains("past the cache capacity"));
    assert!(c.check_set(4, 0, 100).is_err() && c.check_set(0, 100, 100).is_err());
}
