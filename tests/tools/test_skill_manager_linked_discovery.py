"""Public skill_manage contracts for configured symlink-mounted skill packages."""
import json
import sys

import pytest

from agent import skill_utils
from tools import skill_manager_tool as manager

CONTENT = "---\nname: target\ndescription: Use when testing linked skills.\n---\n\nBefore marker.\n"
pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="Directory symlinks require Windows privileges")


@pytest.fixture
def roots(tmp_path, monkeypatch):
    local = tmp_path / "active" / "skills"
    local.mkdir(parents=True)
    monkeypatch.setattr(manager, "SKILLS_DIR", local)
    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [local])
    return local, tmp_path / "shared"


def package(path):
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(CONTENT, encoding="utf-8")
    return path


def patch_skill(name="target", **changes):
    op = {"action": "patch", "name": name, "old_string": "Before marker.", "new_string": "After marker."}
    op.update(changes)
    return op


def apply(*operations):
    return json.loads(manager.skill_manage(action="", name="", operations=list(operations)))


def test_patch_mounted_skill_updates_shared_target_without_copy(roots):
    local, shared = roots
    target = package(shared / "target")
    link = local / "target"
    link.symlink_to(target, target_is_directory=True)
    result = apply(patch_skill())
    assert result["success"] is True, result
    assert (target / "SKILL.md").read_text() == CONTENT.replace("Before marker.", "After marker.")
    assert link.is_symlink()
    assert link.resolve() == target


def test_patch_mount_category_by_categorized_name(roots):
    local, shared = roots
    target = package(shared / "finance" / "target")
    (local / "finance").symlink_to(shared / "finance", target_is_directory=True)
    result = apply(patch_skill("finance/target"))
    assert result["success"] is True, result
    assert "After marker." in (target / "SKILL.md").read_text()
    assert (local / "finance").is_symlink()


def test_patch_mounted_skill_in_configured_external_root(roots, tmp_path, monkeypatch):
    local, shared = roots
    external = tmp_path / "external"
    external.mkdir()
    target = package(shared / "target")
    (external / "target").symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(skill_utils, "get_all_skills_dirs", lambda: [local, external])
    result = apply(patch_skill())
    assert result["success"] is True, result
    assert "After marker." in (target / "SKILL.md").read_text()
    assert not (local / "target").exists()


def test_regular_local_skill_still_patches(roots):
    local, _ = roots
    target = package(local / "target")
    result = apply(patch_skill())
    assert result["success"] is True, result
    assert "After marker." in (target / "SKILL.md").read_text()


@pytest.mark.parametrize("excluded", [".git", ".archive"])
def test_excluded_mounted_package_is_not_mutated(roots, excluded):
    local, shared = roots
    hidden = local / excluded
    hidden.mkdir()
    target = package(shared / "target")
    (hidden / "target").symlink_to(target, target_is_directory=True)
    result = apply(patch_skill())
    assert result["success"] is False, result
    assert (target / "SKILL.md").read_text() == CONTENT


def test_support_directory_fake_skill_is_not_mutated(roots):
    local, _ = roots
    parent = package(local / "parent")
    (parent / "SKILL.md").write_text(CONTENT.replace("name: target", "name: parent"))
    target = package(local / "parent" / "scripts" / "target")
    result = apply(patch_skill())
    assert result["success"] is False, result
    assert (target / "SKILL.md").read_text() == CONTENT


def test_unconfigured_other_profile_remains_untouched(roots, tmp_path):
    local, shared = roots
    target = package(shared / "target")
    (local / "target").symlink_to(target, target_is_directory=True)
    other = package(tmp_path / "other-profile" / "skills" / "target")
    result = apply(patch_skill())
    assert result["success"] is True, result
    assert "After marker." in (target / "SKILL.md").read_text()
    assert (other / "SKILL.md").read_text() == CONTENT


def test_failed_batch_restores_real_target_and_preserves_mount(roots):
    local, shared = roots
    target = package(shared / "target")
    link = local / "target"
    link.symlink_to(target, target_is_directory=True)
    result = apply(patch_skill(), patch_skill(old_string="Missing marker.", new_string="Not written."))
    assert result["success"] is False, result
    assert (target / "SKILL.md").read_text() == CONTENT
    assert link.is_symlink()
    assert link.resolve() == target
    assert not list(shared.glob("*.rollback-broken"))


def test_unconfigured_mount_is_not_discovered(roots, tmp_path):
    _, shared = roots
    target = package(shared / "target")
    unrelated = tmp_path / "other-profile" / "skills"
    unrelated.mkdir(parents=True)
    (unrelated / "target").symlink_to(target, target_is_directory=True)
    result = apply(patch_skill())
    assert result["success"] is False, result
    assert (target / "SKILL.md").read_text() == CONTENT
