# Regeln für Arbeiten an diesem Repository

- Commits enthalten keine Claude-Attribution: keine `Co-Authored-By`-Zeile, keine
  „Generated with“-Zeile, kein Emoji-Zusatz. Das gilt auch für Pull-Request-Texte.
- Gepushte Commits auf `main` werden nie per amend oder force-push umgeschrieben.
  Nachträge sind immer ein neuer Commit.
- Was nicht gegen eine Quelle oder einen Lauf geprüft ist, wird im Code mit einem
  Kommentar `UNGEPRUEFT` markiert, und der Selbsttest sichert es ab, wo das geht.
- Jede Aussage in Berichten trägt ihre Signalstärke: `[gemessen]`, `[belegt]`,
  `[abgeleitet]` oder `[geschaetzt]`.
- Änderungen werden danach beurteilt, wie sie sich beim späteren Betrieb der
  Pipeline auswirken: ein Befehl, wenig Eingriffe, lesbare Hörliste.
- Schritte, Anleitungen und Befehlsfolgen stehen in Ausführungsreihenfolge.
- Keine persönlichen Daten, keine echten Interviewdaten und keine Pfade echter
  Aufnahmen im Repository. Tests arbeiten nur mit synthetischen Daten.
- `diar2_merge.py` bleibt reine Standardbibliothek. Vor jedem Commit:
  `python -m pytest tests` und `shellcheck -s bash diar2.sh install_mac.sh selftest_mac.sh`.
