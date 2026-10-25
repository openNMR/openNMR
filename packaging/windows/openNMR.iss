; Inno Setup script: dist\openNMR (PyInstaller's folder) -> a normal
; Windows installer, openNMR-<version>-windows-setup.exe.
;
; Built by packaging/make_installer.py (which passes the defines below);
; by hand:  iscc /DAppVersion=0.2.0 packaging\windows\openNMR.iss
;
; * Per-user install, no administrator rights needed (PrivilegesRequired=
;   lowest -> %LOCALAPPDATA%\Programs\openNMR); an administrator can still
;   choose "all users" in the first dialog.
; * Start-menu entry, optional desktop icon, an uninstaller in Settings >
;   Apps.
; * Running a newer setup upgrades in place: the same AppId is the same
;   application. The old version's program files are removed first
;   ([InstallDelete]), so a module deleted from openNMR cannot linger.
;   User settings live in %USERPROFILE%\.opennmr and are never touched.
; * NEVER change AppId: a new one makes Windows treat the next version as a
;   different program, installed beside the old one.

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\openNMR"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

[Setup]
AppId={{8F3C5B1E-6A2D-4C7E-9B41-2E5D7A0C9F13}
AppName=openNMR
AppVersion={#AppVersion}
AppVerName=openNMR {#AppVersion}
AppPublisher=openNMR developers
VersionInfoVersion={#AppVersion}
DefaultDirName={autopf}\openNMR
DefaultGroupName=openNMR
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutputDir}
OutputBaseFilename=openNMR-{#AppVersion}-windows-setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64
UninstallDisplayIcon={app}\openNMR.exe
CloseApplications=yes
#ifdef IconFile
SetupIconFile={#IconFile}
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\openNMR"; Filename: "{app}\openNMR.exe"
Name: "{autodesktop}\openNMR"; Filename: "{app}\openNMR.exe"; Tasks: desktopicon

[Run]
; A normal install ends with the usual "Launch openNMR" tick-box (never in a
; silent install, so CI's silent install does not open a window) ...
Filename: "{app}\openNMR.exe"; Description: "{cm:LaunchProgram,openNMR}"; Flags: nowait postinstall skipifsilent
; ... and the in-app updater (opennmr.update_install) runs this setup silently
; with /RELAUNCH=1, which starts the new version again once it is in place.
Filename: "{app}\openNMR.exe"; Flags: nowait; Check: RelaunchRequested

[Code]
function RelaunchRequested: Boolean;
begin
  Result := ExpandConstant('{param:RELAUNCH|0}') = '1';
end;
