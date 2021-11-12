#ifndef TACO_MODE_FORMAT_LG_SINGLETON_H
#define TACO_MODE_FORMAT_LG_SINGLETON_H

#include "taco/lower/mode_format_impl.h"

namespace taco {

class LgSingletonModeFormat : public ModeFormatImpl {
public:
  LgSingletonModeFormat();
  LgSingletonModeFormat(bool isFull, bool isOrdered,
                      bool isUnique, bool isZeroless, long long allocSize = DEFAULT_ALLOC_SIZE);

  ~LgSingletonModeFormat() override {}

  ModeFormat copy(std::vector<ModeFormat::Property> properties) const override;

  ModeFunction posIterBounds(ir::Expr parentPos, Mode mode) const override;
  ModeFunction posIterAccess(ir::Expr pos, std::vector<ir::Expr> coords,
                             Mode mode) const override;

  std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, int level) const override;
  std::vector<ModeRegion> getRegions(ir::Expr tensor, int level) const override;

private:
  enum RECT_COMPRESSED_REGIONS {
    CRD,
    CRD_PARENT,
  };
  ir::Expr getRegion(ModePack pack, RECT_COMPRESSED_REGIONS reg) const;
  ir::Expr getAccessor(ModePack pack, RECT_COMPRESSED_REGIONS reg, ir::RegionPrivilege priv = ir::RO) const;

  static inline ir::Expr fidCoord = ir::Symbol::make("FID_COORD");
  const long long allocSize;
};

}

#endif //TACO_MODE_FORMAT_LG_SINGLETON_H
